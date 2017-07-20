require 'gnuplot'

function convert(csv)
    local list = {}
    for value in (csv .. ","):gmatch("(%S+)%W*,") do table.insert(list,tonumber(value))
    print(value,"value") end
    return unpack(list)
end

local file = io.open("logs/train1500551633.2307.log")
i=0
x=torch.Tensor()
if file then
	for line in file:lines() do
		er1,er2=unpack(line:split(" "))
		if(i>0) then
			er1=convert(er1)
			x=torch.cat(x,torch.Tensor({er1}));
		end	
		i=i+1
	end
end	

gnuplot.pngfigure('logs/train.png')
gnuplot.plot({'Train error',x})
gnuplot.xlabel('iterations')
gnuplot.ylabel('error')
gnuplot.plotflush()
