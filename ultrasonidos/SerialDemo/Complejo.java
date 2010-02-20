import java.util.*;
import java.math.*;

public class Complejo {

public Complejo(int ancho1, int ancho2, int x, int y)
{Recta[] cuadrado = new Recta[4];

 Posicion A = new Posicion(0, 0);
 Posicion B = new Posicion(0, 500);
 Posicion C = new Posicion(500, 500);
 Posicion D = new Posicion(ancho1, 0);
 Posicion E = new Posicion(ancho1, 500 - ancho2);
 Posicion F = new Posicion(500, 500 - ancho2);
 


 cuadrado[0]=new Recta(A,B);
 cuadrado[1]=new Recta(B,C);
 cuadrado[2]=new Recta(D,E);
 cuadrado[3]=new Recta(E,F);

Posicion posicionCarro = new Posicion(x, y);

 double orientacionUltra =  0.5 * Math.PI; /* radianes */

 orientacionUltra = orientacionUltra % (2*Math.PI);
 double arcoUltra = 15; /* grados */
 double arcoRad = 15 * Math.PI / 180;

 double orientacionCarro = Math.PI /2; /*radianes */
 double ancho = 12;
 double largo = 20; 

 double volante = 0;
 Carro carro = new Carro(posicionCarro, ancho, largo, orientacionCarro);
 carro.fijaVolante(volante);
Posicion posicionUltra1 = new Posicion(10,6);
 Posicion posicionUltra2 = new Posicion(10, 2);
 Posicion posicionUltra3 = new Posicion(10, -2);
 Posicion posicionUltra4 = new Posicion(10, -6);
 Posicion posicionUltra5 = new Posicion(8, 6);
 Posicion posicionUltra6 = new Posicion(8, -6);
 Posicion posicionUltra7 = new Posicion(-3, 6);
 Posicion posicionUltra8 = new Posicion(-3, -6);
 Posicion posicionUltra9 = new Posicion(-10, 6);
 Posicion posicionUltra10 = new Posicion(-10, -6);
 Posicion posicionUltra11 = new Posicion(-10, 0);



 Posicion posicionUltraReal1 = new Posicion(-posicionUltra1.y*Math.sin(orientacionCarro) + posicionUltra1.x*Math.cos(orientacionCarro)+ posicionCarro.x, posicionUltra1.x*Math.sin(orientacionCarro) + posicionUltra1.y*Math.cos(orientacionCarro)+ posicionCarro.y);
 Posicion posicionUltraReal2 = new Posicion(-posicionUltra2.y*Math.sin(orientacionCarro) + posicionUltra2.x*Math.cos(orientacionCarro)+ posicionCarro.x, posicionUltra2.x*Math.sin(orientacionCarro) + posicionUltra2.y*Math.cos(orientacionCarro)+ posicionCarro.y);
 Posicion posicionUltraReal3 = new Posicion(-posicionUltra3.y*Math.sin(orientacionCarro) + posicionUltra3.x*Math.cos(orientacionCarro)+ posicionCarro.x, posicionUltra3.x*Math.sin(orientacionCarro) + posicionUltra3.y*Math.cos(orientacionCarro)+ posicionCarro.y);
 Posicion posicionUltraReal4 = new Posicion(-posicionUltra4.y*Math.sin(orientacionCarro) + posicionUltra4.x*Math.cos(orientacionCarro)+ posicionCarro.x, posicionUltra4.x*Math.sin(orientacionCarro) + posicionUltra4.y*Math.cos(orientacionCarro)+ posicionCarro.y);
 Posicion posicionUltraReal5 = new Posicion(-posicionUltra5.y*Math.sin(orientacionCarro) + posicionUltra5.x*Math.cos(orientacionCarro)+ posicionCarro.x, posicionUltra5.x*Math.sin(orientacionCarro) + posicionUltra5.y*Math.cos(orientacionCarro)+ posicionCarro.y);
 Posicion posicionUltraReal6 = new Posicion(-posicionUltra6.y*Math.sin(orientacionCarro) + posicionUltra6.x*Math.cos(orientacionCarro)+ posicionCarro.x, posicionUltra6.x*Math.sin(orientacionCarro) + posicionUltra6.y*Math.cos(orientacionCarro)+ posicionCarro.y);
 Posicion posicionUltraReal7 = new Posicion(-posicionUltra7.y*Math.sin(orientacionCarro) + posicionUltra7.x*Math.cos(orientacionCarro)+ posicionCarro.x, posicionUltra7.x*Math.sin(orientacionCarro) + posicionUltra7.y*Math.cos(orientacionCarro)+ posicionCarro.y);
 Posicion posicionUltraReal8 = new Posicion(-posicionUltra8.y*Math.sin(orientacionCarro) + posicionUltra8.x*Math.cos(orientacionCarro)+ posicionCarro.x, posicionUltra8.x*Math.sin(orientacionCarro) + posicionUltra8.y*Math.cos(orientacionCarro)+ posicionCarro.y);
 Posicion posicionUltraReal9 = new Posicion(-posicionUltra9.y*Math.sin(orientacionCarro) + posicionUltra9.x*Math.cos(orientacionCarro)+ posicionCarro.x, posicionUltra9.x*Math.sin(orientacionCarro) + posicionUltra9.y*Math.cos(orientacionCarro)+ posicionCarro.y);
 Posicion posicionUltraReal10 = new Posicion(-posicionUltra10.y*Math.sin(orientacionCarro) + posicionUltra10.x*Math.cos(orientacionCarro)+ posicionCarro.x, posicionUltra10.x*Math.sin(orientacionCarro) + posicionUltra10.y*Math.cos(orientacionCarro)+ posicionCarro.y);
 Posicion posicionUltraReal11 = new Posicion(-posicionUltra11.y*Math.sin(orientacionCarro) + posicionUltra11.x*Math.cos(orientacionCarro)+ posicionCarro.x, posicionUltra11.x*Math.sin(orientacionCarro) + posicionUltra11.y*Math.cos(orientacionCarro)+ posicionCarro.y);

  
Ultrasonido[] ultras = new Ultrasonido[11];
 ultras[0]= new Ultrasonido(posicionUltraReal1, (Math.PI/6 + orientacionCarro) % (2*Math.PI), arcoRad);
 ultras[0].fijaPosicionRel(posicionUltra1);
 ultras[1]= new Ultrasonido(posicionUltraReal2, (0 + orientacionCarro)% (2*Math.PI), arcoRad);
 ultras[1].fijaPosicionRel(posicionUltra2);
 ultras[2]= new Ultrasonido(posicionUltraReal3, (0 + orientacionCarro)% (2*Math.PI), arcoRad);
 ultras[2].fijaPosicionRel(posicionUltra3);
 ultras[3]= new Ultrasonido(posicionUltraReal4,  (11*Math.PI/6 + orientacionCarro) % (2*Math.PI), arcoRad);
 ultras[3].fijaPosicionRel(posicionUltra4);
 ultras[4]= new Ultrasonido(posicionUltraReal5, (Math.PI/2 + orientacionCarro) % (2*Math.PI), arcoRad);
 ultras[4].fijaPosicionRel(posicionUltra5);
 ultras[5]= new Ultrasonido(posicionUltraReal6, (-Math.PI/2 + orientacionCarro) % (2*Math.PI), arcoRad);
 ultras[5].fijaPosicionRel(posicionUltra6);
 ultras[6]= new Ultrasonido(posicionUltraReal7, (Math.PI/2 + orientacionCarro) % (2*Math.PI), arcoRad);
 ultras[6].fijaPosicionRel(posicionUltra7);
 ultras[7]= new Ultrasonido(posicionUltraReal8, (-Math.PI/2 + orientacionCarro) % (2*Math.PI), arcoRad);
 ultras[7].fijaPosicionRel(posicionUltra8);
 ultras[8]= new Ultrasonido(posicionUltraReal9,  (3*Math.PI/4 + orientacionCarro) % (2*Math.PI), arcoRad);
 ultras[8].fijaPosicionRel(posicionUltra9);
 ultras[9]= new Ultrasonido(posicionUltraReal10, (-3*Math.PI/4 + orientacionCarro) % (2*Math.PI), arcoRad);
 ultras[9].fijaPosicionRel(posicionUltra10);
 ultras[10]= new Ultrasonido(posicionUltraReal11, (Math.PI + orientacionCarro) % (2*Math.PI), arcoRad);
 ultras[10].fijaPosicionRel(posicionUltra11);




for (int l=0; l<ultras.length; l++)
{
double[] distanciaTotal = new double[cuadrado.length];
 for (int i =0; i < cuadrado.length; i++)
 {
   double[] distanciasRecta = new double[5];

 Posicion[] puntos = new Posicion[5];
 puntos[0]= cuadrado[i].posicionInicio;
 puntos[1]= cuadrado[i].posicionFinal;
 double cuadradoA = cuadrado[i].dimeA();
 double cuadradoB = cuadrado[i].dimeB();
 double x0 = ultras[l].dimePosicion().x;
 double y0 = ultras[l].dimePosicion().y;
 puntos[2] = cuadrado[i].dimeCorte(new Recta(cuadradoB, -cuadradoA, cuadradoA*y0 - cuadradoB*x0));
 double anguloMayor = (ultras[l].orientacion + ultras[l].arco + 0.02) ;
 double anguloMenor = ultras[l].orientacion - ultras[l].arco - 0.02;
 double tangentePos = Math.tan(ultras[l].orientacion + ultras[l].arco);
 puntos[3] = cuadrado[i].dimeCorte(new Recta(tangentePos, -1, -tangentePos*x0 + y0));
 tangentePos = Math.tan(ultras[l].orientacion - ultras[l].arco);
 puntos[4] = cuadrado[i].dimeCorte(new Recta(tangentePos, -1, -tangentePos*x0 + y0));
if (l==3 && i==7) 
{System.out.println("Recta:" + i); 
 System.out.println("   p0=" + puntos[0].x + "," + puntos[0].y);
 System.out.println("   p1=" + puntos[1].x + "," + puntos[1].y);
 System.out.println("   p2=" + puntos[2].x + "," + puntos[2].y);
 System.out.println("   p3=" + puntos[3].x + "," + puntos[3].y);
 System.out.println("   p4=" + puntos[4].x + "," + puntos[4].y);



}


 for (int j=0; j < puntos.length; j++)
  {//System.out.println("(" + puntos[j].x + "," + puntos[j].y + ")" + ultras[l].dimePosicion().dimeAngulo(puntos[j])); 
   double anguloFormado =  ultras[l].dimePosicion().dimeAngulo(puntos[j]);
if ( (((puntos[j].x  - puntos[0].x) * (puntos[j].x  - puntos[1].x)) <= 0) && (((puntos[j].y  - puntos[0].y) * (puntos[j].y  - puntos[1].y)) <= 0)  )
{   if (anguloMenor >= 0)
    {if ( ((anguloFormado >= anguloMenor) && (anguloFormado <= anguloMayor))  )
     {
      distanciasRecta[j]= ultras[l].dimePosicion().dimeDistancia(puntos[j]);
if (l==3 && i==7)
{System.out.println(j + "  * Caso 1:" + distanciasRecta[j]);}

     }
     else
     {//System.out.println("Fuera del arco");
      distanciasRecta[j]= 100000;
if (l==3 && i==7)
{System.out.println(j + "  * Caso 2:" + distanciasRecta[j]);
 System.out.println(anguloMenor + "," + anguloFormado + "," + anguloMayor);
}

     }
    }
   if (anguloMenor < 0)
   {if ( (anguloFormado >= (anguloMenor+ 2*Math.PI)) || (anguloFormado <= anguloMayor) )
     {//System.out.println("Dentro del arco");
      distanciasRecta[j]= ultras[l].dimePosicion().dimeDistancia(puntos[j]);
if (l==3 && i==7)
{System.out.println(j + "  * Caso 3:" + distanciasRecta[j]);}

     }
     else
     {//System.out.println("Fuera del arco");
      distanciasRecta[j]= 100000;
if (l==3 && i==7)
{System.out.println(j + "  * Caso 4:" + distanciasRecta[j]);}

     }
   }
}
else
{
if (l==3 && i==7)
{System.out.println("**************");}
 
distanciasRecta[j]= 100000;
}

   }
 double distanciaMinimaRecta = 100000; 
 for (int k=0; k< puntos.length; k++)
  {if (distanciasRecta[k] < distanciaMinimaRecta)
   {distanciaMinimaRecta = distanciasRecta[k];
   } 
  }  
 
distanciaTotal[i]= distanciaMinimaRecta;
}
double distanciaGlobal = 1000000;
for (int k=0; k<distanciaTotal.length; k++)
{//System.out.println("Distancias = " + distanciaTotal[k]);
 if (distanciaGlobal > distanciaTotal[k])
 {distanciaGlobal = distanciaTotal[k];
 }
}
ultras[l].fijaRadio(distanciaGlobal);
//System.out.println("Distancia Global = " + distanciaGlobal);

if (l==3)
{for (int k=0; k<distanciaTotal.length; k++)
{System.out.println("Distancias[" + k + "] = " + distanciaTotal[k]);
 
}

}

}

Plano plano = new Plano(cuadrado, ultras, carro);
plano.setSize(900,600);
plano.setVisible(true);

}



public static void main(String[] args) {
int anchoEnt = (int)(Integer.parseInt(args[0]));
int anchoSal = (int)(Integer.parseInt(args[1]));
int x = (int)(Integer.parseInt(args[2]));
int y = (int)(Integer.parseInt(args[3]));

Complejo u = new Complejo(anchoEnt, anchoSal, x, y);
}


}