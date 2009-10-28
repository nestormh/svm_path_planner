import java.util.*;
import java.math.*;

public class Visual {

public Visual()
{

 
int x = 300;
int y = 300;
Posicion posicionCarro = new Posicion(x, y);

 double orientacionUltra =  0.5 * Math.PI; /* radianes */

 orientacionUltra = orientacionUltra % (2*Math.PI);
 double arcoUltra = 15; /* grados */
 double arcoRad = 15 * Math.PI / 180;

 double orientacionCarro = Math.PI /2; /*radianes */
 double ancho = 400;
 double largo = 300; 

 double volante = 0;
 Carro carro = new Carro(posicionCarro, ancho, largo, orientacionCarro);
 carro.fijaVolante(volante);
//delante
Posicion posicionUltra1 = new Posicion(150,200);
 Posicion posicionUltra2 = new Posicion(150,160);
 Posicion posicionUltra3 = new Posicion(150, 75);
 Posicion posicionUltra4 = new Posicion(150, 35);
//laterales
 Posicion posicionUltra5 = new Posicion(150, -35);
 Posicion posicionUltra6 = new Posicion(150, -75);
 Posicion posicionUltra7 = new Posicion(150, -160);
 Posicion posicionUltra8 = new Posicion(150, -200);
//traseros
 Posicion posicionUltra9 = new Posicion(-100, 75);
 Posicion posicionUltra10 = new Posicion(-100, -75);
 Posicion posicionUltra11 = new Posicion(-100, 0);



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
 ultras[0]= new Ultrasonido(posicionUltraReal1, (0 + orientacionCarro) % (2*Math.PI), arcoRad);
 ultras[0].fijaPosicionRel(posicionUltra1);
 ultras[1]= new Ultrasonido(posicionUltraReal2, (0 + orientacionCarro)% (2*Math.PI), arcoRad);
 ultras[1].fijaPosicionRel(posicionUltra2);
 ultras[2]= new Ultrasonido(posicionUltraReal3, (0 + orientacionCarro)% (2*Math.PI), arcoRad);
 ultras[2].fijaPosicionRel(posicionUltra3);
 ultras[3]= new Ultrasonido(posicionUltraReal4,  (0 + orientacionCarro) % (2*Math.PI), arcoRad);
 ultras[3].fijaPosicionRel(posicionUltra4);
 ultras[4]= new Ultrasonido(posicionUltraReal5, (0 + orientacionCarro) % (2*Math.PI), arcoRad);
 ultras[4].fijaPosicionRel(posicionUltra5);
 ultras[5]= new Ultrasonido(posicionUltraReal6, (0 + orientacionCarro) % (2*Math.PI), arcoRad);
 ultras[5].fijaPosicionRel(posicionUltra6);
 ultras[6]= new Ultrasonido(posicionUltraReal7, (0 + orientacionCarro) % (2*Math.PI), arcoRad);
 ultras[6].fijaPosicionRel(posicionUltra7);
 ultras[7]= new Ultrasonido(posicionUltraReal8, (0  + orientacionCarro) % (2*Math.PI), arcoRad);
 ultras[7].fijaPosicionRel(posicionUltra8);
 ultras[8]= new Ultrasonido(posicionUltraReal9,  (3*Math.PI/4 + orientacionCarro) % (2*Math.PI), arcoRad);
 ultras[8].fijaPosicionRel(posicionUltra9);
 ultras[9]= new Ultrasonido(posicionUltraReal10, (-3*Math.PI/4 + orientacionCarro) % (2*Math.PI), arcoRad);
 ultras[9].fijaPosicionRel(posicionUltra10);
 ultras[10]= new Ultrasonido(posicionUltraReal11, (Math.PI + orientacionCarro) % (2*Math.PI), arcoRad);
 ultras[10].fijaPosicionRel(posicionUltra11);

int radio = 200;

for (int j=0; j<ultras.length; j++)
{ultras[j].fijaRadio(radio);
}

//int id0 = 7;
//int id1 = 8;
//int id2 = 5;
//int id3 = 13;
//int id4 = 4;
//int id5 = 16;
//int id6 = 12;
//int id7 = 11;


int id0 = 11;
int id1 = 12;
int id2 = 16;
int id3 = 4;
int id4 = 13;
int id5 = 5;
int id6 = 8;
int id7 = 7;

int id8 = 15;
int id9 = 15;
int id10 = 15;

ultras[7].fijaIdentificador(id0);
ultras[6].fijaIdentificador(id1);
ultras[5].fijaIdentificador(id2);
ultras[4].fijaIdentificador(id3);
ultras[3].fijaIdentificador(id4);
ultras[2].fijaIdentificador(id5);
ultras[1].fijaIdentificador(id6);
ultras[0].fijaIdentificador(id7);


ultras[8].fijaIdentificador(id8);
ultras[9].fijaIdentificador(id9);
ultras[10].fijaIdentificador(id10);



plano = new PlanoVisual(ultras, carro);
plano.setSize(900,700);
plano.setVisible(true);

//para probar
//fijaRadio(2,120,120);
}

PlanoVisual plano;

public static void main(String[] args) {
Visual u = new Visual();
}

public void fijaRadio(int i, int med1, int med2)
{try{
 plano.actualizar(i, med1, med2);
 } catch (Exception e) {};
}


}