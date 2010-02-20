
/* dibujo de la gráfica del error 
*/

/* importar las librerías gráficas */
import java.awt.*;
import java.awt.event.*;
import java.applet.*;
import java.lang.*;

/* la clase dibujo extiende la clase Canvas */

/**
* Dibujo de la gráfica, dado un vector
* @author Evelio José González González
* @version 13-11-99
*/

public class DibujoVisual extends Canvas 
{
/* variables globales vector (donde se alamacena el error) */
/* y el máximo del vector, para poder dibujar a escala */


public Ultrasonido[] ultras;
public Carro carro;
/**
* constructor que acepta como argumento el vector a dibujar
*/
 public DibujoVisual(Ultrasonido[] ultras, Carro carro) {    
/* se iguala la variable local vector a la variable */
/* pasada como argumento datos */ 		
 this.ultras=ultras;
 this.carro = carro;
/* se delimita el tamaño en pixels del dibujo */ 
 setSize(600,600);
/* mostrar el dibujo */
 setVisible(true); 
 }

/**
* actualización del dibujo 
* llama al método paint, evitando asi parpadeos 
* de la pantalla 
*/
 public synchronized void update(Graphics g)
 {paint(g);}

/**
* pinta la gráfica del vector 
*/
 public synchronized void paint(Graphics g)
 {   Color t = new Color(210,210,255);
 Dimension s= getSize();
 g.setColor(t);
 g.fillRect(0,0,s.width, s.height); 
 g.setColor(Color.black);
 /* dibujar carro */
 g.setColor(Color.blue);
 int Ax = (int)(carro.dimePosicion().x + (0.5 * carro.largo * Math.cos(carro.orientacion)) - (0.5 * carro.ancho * Math.sin(carro.orientacion))); 
 int Ay =  (int)(carro.dimePosicion().y + (0.5 * carro.largo * Math.sin(carro.orientacion)) + (0.5 * carro.ancho * Math.cos(carro.orientacion)));
 int Bx = (int)(Ax - carro.largo * Math.cos(carro.orientacion)); 
 int By =  (int)(Ay - carro.largo * Math.sin(carro.orientacion) );
 int Cx = (int)(Ax + carro.ancho * Math.sin(carro.orientacion)); 
 int Cy =  (int)(Ay - carro.ancho * Math.cos(carro.orientacion) );
 int Dx = (int)(Cx - carro.largo * Math.cos(carro.orientacion)); 
 int Dy =  (int)(Cy - carro.largo * Math.sin(carro.orientacion) );

 g.drawLine(Ax, s.height - Ay ,Bx, s.height - By ); 
 g.drawLine(Ax, s.height - Ay ,Cx, s.height - Cy ); 
 g.drawLine(Bx, s.height - By ,Dx, s.height - Dy ); 
 g.drawLine(Cx, s.height - Cy ,Dx, s.height - Dy ); 

 g.setColor(Color.blue);
 g.drawLine(Ax +1 , s.height - Ay ,Cx +1 , s.height - Cy ); 


 for (int i=0; i<ultras.length; i++)
 {double centrox = ultras[i].x;
  double centroy = ultras[i].y;
  //System.out.println("Y=" + centroy + "****" + Ay);
  double radio = ultras[i].radio;

  double rojo = 200;
 double inicioXrojo = centrox - rojo;
  double inicioYrojo = s.height - centroy - rojo;
  
  double inicioX = centrox - radio;
  double inicioY = s.height - centroy - radio;
  double orientacion = ultras[i].orientacion;
  double arco = ultras[i].arco;
g.setColor(Color.red);
g.fillArc((int)inicioXrojo ,(int)inicioYrojo , (int)(2*rojo), (int)(2*rojo), (int)((orientacion - arco)*180/Math.PI), (int)((2*arco)*180/Math.PI));
 

g.setColor(Color.green);
  
//escalado; 
 g.fillArc((int)inicioX ,(int)inicioY , (int)(2*radio), (int)(2*radio), (int)((orientacion - arco)*180/Math.PI), (int)((2*arco)*180/Math.PI));
// g.setColor(Color.black);
 //g.drawString(String.valueOf(radio),10, 15*(i+1));
 }
/* las ruedas */
g.setColor(Color.black);

double pBx = carro.x + 0.5 * carro.largo * Math.cos(carro.orientacion) - 0.5 * carro.ancho * Math.sin(carro.orientacion);
double pBy = carro.y + 0.5 * carro.largo * Math.sin(carro.orientacion) + 0.5 * carro.ancho * Math.cos(carro.orientacion);
 
double pBx1 = pBx + 0.25 * carro.largo* Math.cos(carro.volante + carro.orientacion);
double pBy1 = pBy + 0.25 * carro.largo* Math.sin(carro.volante + carro.orientacion);

double pBx2 = pBx - 0.25 * carro.largo* Math.cos(carro.volante + carro.orientacion);
double pBy2 = pBy - 0.25 * carro.largo* Math.sin(carro.volante + carro.orientacion);

 g.drawLine((int)pBx1, s.height - (int)pBy1 , (int)pBx2 , s.height - (int)pBy2 ); 

double pCx = carro.x + 0.5 * carro.largo * Math.cos(carro.orientacion) + 0.5 * carro.ancho * Math.sin(carro.orientacion);
double pCy = carro.y + 0.5 * carro.largo * Math.sin(carro.orientacion) - 0.5 * carro.ancho * Math.cos(carro.orientacion);
 
double pCx1 = pCx + 0.25 * carro.largo* Math.cos(carro.volante + carro.orientacion);
double pCy1 = pCy + 0.25 * carro.largo* Math.sin(carro.volante + carro.orientacion);

double pCx2 = pCx - 0.25 * carro.largo* Math.cos(carro.volante + carro.orientacion);
double pCy2 = pCy - 0.25 * carro.largo* Math.sin(carro.volante + carro.orientacion);

 g.drawLine((int)pCx1, s.height - (int)pCy1 , (int)pCx2 , s.height - (int)pCy2 ); 


/*  System.out.println(pBx1 + "," + pBy1 + "---" +  pBx2 + "," +  pBy2 );  */
 

 }

}  
