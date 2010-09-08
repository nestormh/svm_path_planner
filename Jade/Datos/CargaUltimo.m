#Cargamos ultimo fichero de datos
[st,out]=system("ls PruPanLog*.mat | tail -1");
if(st!=0)
   error("Al obtener lista de fichero de datos");
else
   disp(["Cargando el fichero ", out])
   load(out(1:length(out)-1))
endif