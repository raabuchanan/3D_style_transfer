require 'nn'
require 'cunn'
require 'image'
require 'csvigo'
require'lfs'
require 'image'
require 'torch'
require 'lsqlite3'
require 'json'
--require 'torchffi'


function split(str)
    tbl = {}
    for c in str:gmatch('.') do
         table.insert(tbl, c)
    end
    return tbl
end

-- db=sqlite3.open('style_cup_194.db')  -- open
db=sqlite3.open('glove_mosaic_learned.db')  -- open

-- load model and mean
local data = torch.load( 'models/CNN3_p8_n8_split4_073000.t7' )
local desc = data.desc
local mean = data.mean
local std  = data.std


local image_number = 1

for row in db:nrows("SELECT * FROM descriptors") do
    
	

	if(row.image_id >= 19) then

		local raw_patches = row.data
		local num_key = #row.data/4096;
		local num = row.image_id
		row = nil

		collectgarbage()
		print(num)

		local t = {}
		for i = 1, #raw_patches do
			table.insert(t,string.byte(raw_patches:sub(i, i)))
		    --t[i] = raw_patches:sub(i, i)
		end

		--raw_patches:gsub(".",function(c) table.insert(t,string.byte(c)) end)
		raw_patches = nil
		local d = torch.FloatTensor(t)
		t = nil
		collectgarbage()
		local patches = torch.reshape(d,num_key,1,64,64)
		local output = torch.FloatTensor()

		
		for j=1,patches:size(1) do
		   patches[j] = patches[j]:add( -mean ):cdiv( std )
		end

		local division = math.floor(num_key/10);
		print(division)

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
		

		db:exec[[ CREATE TABLE IF NOT EXISTS learned 
		(image_id INTEGER PRIMARY KEY NOT NULL, 
		rows INTEGER NOT NULL, 
		cols INTEGER NOT NULL, 
		data BLOB,FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE) ]]

		local insert_stmt = db:prepare("INSERT INTO learned (image_id, rows, cols, data) VALUES (?, ?, '128', ? )") 

		insert_stmt:bind(1,num)

		insert_stmt:bind(2,output:size(1))


		local foo = torch.serialize(output:storage())
		local fml = foo:sub(46)



		insert_stmt:bind_blob(3, fml)

		insert_stmt:step()
		insert_stmt:finalize()

		output = nil
		collectgarbage()

	end

	

end

--print("Types: " .. insert_stmt:get_type(1))

db:close()  -- close


