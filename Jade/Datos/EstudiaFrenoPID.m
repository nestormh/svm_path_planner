
#load PruPanLog201006221218.mat
consignaV=controlPID(:,1);
velCS=controlPID(:,2);
deriv=controlPID(:,3);
inte=controlPID(:,4);
comaT=controlPID(:,5);
coma=controlPID(:,6);
apertura=controlPID(:,7);

t=(controlPID_t-controlPID_t(1))/1000;
figure(1)
plot(t,[consignaV velCS inte])

DT=12
#Frenada en llano
T1=147+0.98800-1
ind1=find ((t>T1) & (t<(T1+DT)) );
t1=t(ind1);
t1=t1-t1(1);
l1=length(t1);

consignaV1=consignaV(ind1);
velCS1=velCS(ind1);
deriv1=deriv(ind1);
inte1=inte(ind1);
comaT1=comaT(ind1);
coma1=coma(ind1);
apertura1=apertura(ind1);
IncCom1=[0;coma(2:l1)-coma(1:l1-1)];

figure(2)
plot(t1,consignaV1, ";con;"
     ,t1,velCS1,";vel;"
     ,t1,apertura1,";aper;"
     ,t1, comaT1,";comT;"
     ,t1, coma1,";com;"
#     ,t1, inte1,";int;"
#     ,t1, deriv1,";der;"
   ,t1,consignaV1-velCS1,";err;"
   ,t1,IncCom1,";incC;"
)

#Frenada en bajada
T2=268+1.05900-1
ind2=find ((t>T2) & (t<(T2+DT)) );
t2=t(ind2);
t2=t2-t2(1);
l2=length(t2);

consignaV2=consignaV(ind2);
velCS2=velCS(ind2);
deriv2=deriv(ind2);
inte2=inte(ind2);
comaT2=comaT(ind2);
coma2=coma(ind2);
apertura2=apertura(ind2);
IncCom2=[0;coma(2:l2)-coma(1:l2-1)];

figure(3)
plot(t2,consignaV2, ";con;"
     ,t2,velCS2,";vel;"
     ,t2,apertura2,";aper;"
     ,t2, comaT2,";comT;"
     ,t2, coma2,";com;"
#     ,t2, inte2,";int;"
#     ,t2, deriv2,";der;"
   ,t2,consignaV2-velCS2,";err;"
   ,t2,IncCom2,";incC;"
)

