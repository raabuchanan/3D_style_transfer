require 'nn'
require 'cunn'
require 'image'
require 'lfs'
require 'image'
require 'torch'
require 'lsqlite3'


function split(str)
    tbl = {}
    for c in str:gmatch('.') do
         table.insert(tbl, c)
    end
    return tbl
end

db=sqlite3.open('glove_mosaic_learned.db')  -- load descriptor model

-- load model and mean
local data = torch.load( 'models/CNN3_p8_n8_split4_073000.t7' )
local desc = data.desc
local mean = data.mean
local std  = data.std


local image_number = 1

for row in db:nrows("SELECT * FROM descriptors") do
	local raw_patches = row.data
	local num_key = #row.data/4096;
	local num = row.image_id
	row = nil

	collectgarbage()
	print(num)

	-- First convert patch from string to table
	local t = {}
	for i = 1, #raw_patches do
		table.insert(t,string.byte(raw_patches:sub(i, i)))
	end

	-- Convert table to Tensor and delete table
	raw_patches = nil
	local d = torch.FloatTensor(t)
	t = nil
	collectgarbage()
	local patches = torch.reshape(d,num_key,1,64,64)
	local output = torch.FloatTensor()

	-- Normalize patches
	for j=1,patches:size(1) do
	   patches[j] = patches[j]:add( -mean ):cdiv( std )
	end

	local division = math.floor(num_key/10);
	print(division)

	-- Convert patches to 128D descriptors in blocks depending on number of descriptors
	for ii=1,10 do
		local patches_sec = patches[{{division*(ii-1)+1,ii*division},{},{},{}}]

		-- convert to cuda for processing on the GPU
		patches_sec = patches_sec:cuda()
		desc:cuda()

		-- get descriptor
		local outp = desc:forward( patches_sec ):float()

		output = torch.cat(output,outp,1)
		print(#output)
	end
	
	-- Save learned descriptor to database
	db:exec[[ CREATE TABLE IF NOT EXISTS learned 
	(image_id INTEGER PRIMARY KEY NOT NULL, 
	rows INTEGER NOT NULL, 
	cols INTEGER NOT NULL, 
	data BLOB,FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE) ]]

	local insert_stmt = db:prepare("INSERT INTO learned (image_id, rows, cols, data) VALUES (?, ?, '128', ? )") 

	insert_stmt:bind(1,num)

	insert_stmt:bind(2,output:size(1))


	local raw_string = torch.serialize(output:storage())
	local raw_string_minus_header = raw_string:sub(46)

	insert_stmt:bind_blob(3, raw_string_minus_header)

	insert_stmt:step()
	insert_stmt:finalize()

	output = nil
	collectgarbage()

end

db:close()  -- close


