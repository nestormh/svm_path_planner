consignaV=controlPID(:,1);
velCS=controlPID(:,2);
deriv=controlPID(:,3);
inte=controlPID(:,4);
comaT=controlPID(:,5);
coma=controlPID(:,6);
apertura=controlPID(:,7);
t=(controlPID_t-controlPID_t(1))/1000;
l=length(t);

IncCom=[0;coma(2:l)-coma(1:l-1)];

t=(controlPID_t-controlPID_t(1))/1000;
figure(1)
plot(t,consignaV, ";con;"
     ,t,velCS,";vel;"
     ,t,apertura,";aper;"
     ,t, comaT,";comT;"
     ,t, coma,";com;"
     ,t, inte/50,";int;"
#     ,t, deriv,";der;"
   ,t,consignaV-velCS,";err;"
   ,t,IncCom,";incC;"
)


#Usar otro incremento de comando para frenada
figure(2)
IncCom=[0; coma(2:l)-coma(1:l-1)];
IncCom2=[0; coma(2:l)-comaT(1:l-1)];


plot(t,IncCom,";IncCom;"
#   ,t,IncCom2,";IncCom2;"
     ,t,apertura,";apertura;"
)