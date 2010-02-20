/* marco para la gráfica del error */ 

import java.util.*;
import java.awt.*;

import java.awt.event.*;
import java.applet.*;
import java.lang.*;


/**
* marco para representar la gráfica del error  
* @author Evelio José González González
* @version 15-09-99
*/

public class Plano extends Frame 
 implements WindowListener
 {
/* declaración del botón de cancelar */
/**
* botón de cancelar 
*/
  Button cancelar = new Button ("Grabar");
Button aumentarGiro = new Button ("Volante Izq.");
Button avanzar = new Button ("Avanzar");
Button disminuirGiro = new Button ("Volante Der.");
Button retroceder = new Button ("Retroceder");


/**
* menú para la posibilidad de ampliar la gráfica
*/ 
 MenuItem [] opcionZoom = { new MenuItem("Insertar") };
/**
* barra de menús 
*/
 MenuBar barraMenu = new MenuBar();
 Dibujo graph;
 int inicio = 0;
 Recta[] rectas;
 Ultrasonido[] ultras; 
 Carro carro;
 
  public Plano(Recta[] rectas, Ultrasonido[] ultras, Carro carro){
  this.rectas=rectas;
  this.ultras=ultras;
  this.carro = carro;
/* inicialización del menú */   
/*   for (int i = 0; i<opcionZoom.length;i++)
   {menuZoom.add(opcionZoom[i]);
   }   
   barraMenu.add(menuZoom); */
   setMenuBar(barraMenu);
/* declaración de un campo de texto, donde va a */
/* aparecer el error obtenido en la primera */
/* iteración */  
   TextField texto1 = new TextField("0");
   

/* declaración de un color, que se pone como fondo */
   Color t = new Color(210,210,255);
   setBackground(t);
   addWindowListener(this);
/* dibujo de la gráfica mediante la clase Dibujo, */
/* al que se le pasa como argumento el vector de */
/* error */
   graph=new Dibujo(rectas, ultras, carro);
/* título del marco */
   setTitle("Plano de ultrasonidos");
/* añadir al marco el texto de area, el dibujo */
/* de la gráfica y el botón de cancelar */
 /*  add("North",texto1); */
   add(graph);
/*   add("East",cancelar);*/
/* imprimir el error de la primera iteración */   

/*   texto1.setText("distancias"); */
  cancelar.addActionListener(new Grabar());  
  avanzar.addActionListener(new Avanzar()); 
  retroceder.addActionListener(new AvanzarMenos()); 
  aumentarGiro.addActionListener(new Girar()); 
  disminuirGiro.addActionListener(new GirarMenos()); 

/* declaración de un panel con los campos de texto */
/* y los botones */
  Panel q = new Panel();
  q.setLayout(new GridLayout(1,5));
  q.add(avanzar);
  q.add(retroceder);

  q.add(aumentarGiro);
  q.add(disminuirGiro);
  q.add(cancelar);

  add("South", q); /* añadir el panel al marco */


  }


/**
* método sin implementar
*/
 public void windowIconified(WindowEvent evt)
 {
 }

/**
* método sin implementar
*/
 public void windowDeiconified(WindowEvent evt)
 {
 }

/**
* método sin implementar
*/
 public void windowOpened(WindowEvent evt)
 {
 }

/**
* método sin implementar
*/
 public void windowClosed(WindowEvent evt)
 {
 }

/**
* cerrar el marco sin salir del programa
*/
 public void windowClosing(WindowEvent evt)
 {dispose();
 }

/**
* método sin implementar
*/
 public void windowDeactivated(WindowEvent evt)
 {
 }

/**
* método sin implementar
*/
 public void windowActivated(WindowEvent evt)
 {
 }

/**
* cerrar el marco sin salir del programa
*/
 class Cancelar implements ActionListener {
   public void actionPerformed(ActionEvent e) {
   dispose();
   }
  }

int paso = 2;

public void calculoDistancias()
{
for (int l=0; l<ultras.length; l++)
{
double[] distanciaTotal = new double[rectas.length];
//System.out.println("-----------------");
 for (int i =0; i < rectas.length; i++)
 {
   double[] distanciasRecta = new double[5];

 Posicion[] puntos = new Posicion[5];
 puntos[0]= rectas[i].posicionInicio;
 puntos[1]= rectas[i].posicionFinal;
 double rectasA = rectas[i].dimeA();
 double rectasB = rectas[i].dimeB();
 double x0 = ultras[l].dimePosicion().x;
 double y0 = ultras[l].dimePosicion().y;
 puntos[2] = rectas[i].dimeCorte(new Recta(rectasB, -rectasA, rectasA*y0 - rectasB*x0));
 double anguloMayor = ultras[l].orientacion + ultras[l].arco + 0.015;
 double anguloMenor = ultras[l].orientacion - ultras[l].arco - 0.015;
 double tangentePos = Math.tan(ultras[l].orientacion + ultras[l].arco);
 puntos[3] = rectas[i].dimeCorte(new Recta(tangentePos, -1, -tangentePos*x0 + y0));
 tangentePos = Math.tan(ultras[l].orientacion - ultras[l].arco);
 puntos[4] = rectas[i].dimeCorte(new Recta(tangentePos, -1, -tangentePos*x0 + y0));
 
//System.out.println("**************");
 for (int j=0; j < puntos.length; j++)
  {//System.out.println("(" + puntos[j].x + "," + puntos[j].y + ")" + ultras[l].dimePosicion().dimeAngulo(puntos[j])); 
   double anguloFormado =  ultras[l].dimePosicion().dimeAngulo(puntos[j]);
if ( (((Math.round(puntos[j].x)  - puntos[0].x) * (Math.round(puntos[j].x)  - puntos[1].x)) <= 0) && (((Math.round(puntos[j].y)  - puntos[0].y) * (Math.round(puntos[j].y)  - puntos[1].y)) <= 0)  )
{
  if (anguloMenor >= 0)
    {if ( ((anguloFormado >= anguloMenor) && (anguloFormado <= anguloMayor)) )
     {
//System.out.println("Dentro del arco");

      distanciasRecta[j]= ultras[l].dimePosicion().dimeDistancia(puntos[j]);
//System.out.println("Caso 1: " + distanciasRecta[j]);

     }
     else
     {
//System.out.println("Fuera del arco");
//System.out.println("Caso 4: " + distanciasRecta[j]);

      distanciasRecta[j]= 100000;
     }
    }
   if ((anguloMenor < 0) && (anguloMayor > 0))
   {if ( (anguloFormado >= (anguloMenor+ 2*Math.PI)) || (anguloFormado <= anguloMayor))
     {
 
     distanciasRecta[j]= ultras[l].dimePosicion().dimeDistancia(puntos[j]);
//System.out.println("Caso 2: " + distanciasRecta[j] + "(" + puntos[j].x + "," + puntos[j].y + ")");
//System.out.println(anguloFormado);
//System.out.println(anguloMenor);
//System.out.println(anguloMayor);

     }
     else
     {


//System.out.println("Fuera del arco");
      distanciasRecta[j]= 100000;
//System.out.println("Caso 3: " + distanciasRecta[j]);

     }
   }

   if ((anguloMenor < 0) && (anguloMayor < 0))
   {if ( (anguloFormado >= (anguloMenor+ 2*Math.PI)) && (anguloFormado <= (anguloMayor + 2*Math.PI)))
     {
 
     distanciasRecta[j]= ultras[l].dimePosicion().dimeDistancia(puntos[j]);
//System.out.println("Caso 2: " + distanciasRecta[j] + "(" + puntos[j].x + "," + puntos[j].y + ")");
//System.out.println(anguloFormado);
//System.out.println(anguloMenor);
//System.out.println(anguloMayor);

     }
     else
     {


//System.out.println("Fuera del arco");
      distanciasRecta[j]= 100000;
//System.out.println("Caso 3: " + distanciasRecta[j]);

     }
   }


}
else 
{ distanciasRecta[j]= 100000;
//System.out.println("Caso 5: " + distanciasRecta[j]);


}

 //System.out.println("(" + puntos[j].x + "," + puntos[j].y + ")" + distanciasRecta[j]);

   }
 double distanciaMinimaRecta = 100000; 
 for (int k=0; k< puntos.length; k++)
  {if (distanciasRecta[k] < distanciaMinimaRecta)
   {distanciaMinimaRecta = distanciasRecta[k];
//    System.out.println(distanciasRecta[k]);
   } 
  }  
 
distanciaTotal[i]= distanciaMinimaRecta;
}
double distanciaGlobal = 1000000;
for (int k=0; k<distanciaTotal.length; k++)
{
//System.out.println("Distancias = " + distanciaTotal[k]);
 if (distanciaGlobal > distanciaTotal[k])
 {distanciaGlobal = distanciaTotal[k];

 }
}
ultras[l].fijaRadio(distanciaGlobal);
//System.out.println("Distancia Global = " + distanciaGlobal);
}

 graph.repaint();   
 carro.grabarDato();
   }


class Grabar implements ActionListener {
   public void actionPerformed(ActionEvent e) {
 String cadena = "A" + System.currentTimeMillis();
  carro.volcarDatos(cadena);
  for (int i=0; i < ultras.length; i++)
  {ultras[i].volcarDatos(cadena,i);
  }
 }
}

 class Avanzar implements ActionListener {
   public void actionPerformed(ActionEvent e) {
if (carro.volante == 0)
{carro.x = carro.x + Math.cos(carro.orientacion) * paso;
 carro.y = carro.y + Math.sin(carro.orientacion) * paso;
for(int i=0; i<ultras.length; i++)
{ultras[i].x = ultras[i].x + Math.cos(carro.orientacion) * paso;
 ultras[i].y = ultras[i].y + Math.sin(carro.orientacion) * paso;
}
}
if (carro.volante != 0)
{ double radio = carro.largo / Math.sin(carro.volante);
 /* recta trasera */
 double pAx = carro.x - 0.5 * carro.largo * Math.cos(carro.orientacion);
 double pAy = carro.y - 0.5 * carro.largo * Math.sin(carro.orientacion);
 double tangentePos = - Math.tan(0.5 * Math.PI - carro.orientacion);
 Recta recta1 = new Recta(tangentePos, -1, -tangentePos*pAx + pAy);

 double pBx = carro.x + 0.5 * carro.largo * Math.cos(carro.orientacion) - 0.5 * carro.ancho * Math.sin(carro.orientacion);
 double pBy = carro.y + 0.5 * carro.largo * Math.sin(carro.orientacion) + 0.5 * carro.ancho * Math.cos(carro.orientacion);
 double tangentePos2 = - Math.tan(0.5 * Math.PI - carro.orientacion - carro.volante);
 Recta recta2 = new Recta(tangentePos2, -1, -tangentePos2*pBx + pBy);
 Posicion centro = recta1.dimeCorte(recta2);
//System.out.println("PA:" + pAx + "," + pAy);
//System.out.println("PB:" + pBx + "," + pBy);
//System.out.println(centro.x + "," + centro.y);

 double angulo = centro.dimeAngulo(carro.dimePosicion());

//System.out.println("angulo:" + angulo);
 double radioX = centro.dimeDistancia(carro.dimePosicion()); 

double giros = paso/radioX;

 if (carro.volante < 0 )
 {giros = - giros;
 }

 Posicion B = new Posicion(pBx, pBy);
 double anguloGirado = angulo + giros; 
//System.out.println("anguloGirado:" + anguloGirado);

 carro.x = centro.x + radioX *Math.cos(anguloGirado);
 carro.y = centro.y + radioX *Math.sin(anguloGirado);

/*  carro.orientacion = (anguloGirado + 0.5 * Math.PI) % (2 * Math.PI);*/
  carro.orientacion = (carro.orientacion + giros) % (2 * Math.PI);
for(int i=0; i<ultras.length; i++)
{ultras[i].orientacion = (ultras[i].orientacion + giros)%(2*Math.PI) ;
 ultras[i].x = (carro.dimePosicion().x + ( ultras[i].xRel * Math.cos(carro.orientacion)) - (ultras[i].yRel * Math.sin(carro.orientacion)));
 ultras[i].y = (carro.dimePosicion().y + (ultras[i].xRel * Math.sin(carro.orientacion)) + (ultras[i].yRel * Math.cos(carro.orientacion)));
}

}


calculoDistancias();
}
  }

 class AvanzarMenos implements ActionListener {
   public void actionPerformed(ActionEvent e) {

if (carro.volante == 0)
{carro.x = carro.x - Math.cos(carro.orientacion) * paso;
 carro.y = carro.y - Math.sin(carro.orientacion) * paso;
for(int i=0; i<ultras.length; i++)
{ultras[i].x = ultras[i].x - Math.cos(carro.orientacion) * paso;
 ultras[i].y = ultras[i].y - Math.sin(carro.orientacion) * paso;
}
}
if (carro.volante != 0)
{ double radio = carro.largo / Math.sin(carro.volante);
 /* recta trasera */
 double pAx = carro.x - 0.5 * carro.largo * Math.cos(carro.orientacion);
 double pAy = carro.y - 0.5 * carro.largo * Math.sin(carro.orientacion);
 double tangentePos = - Math.tan(0.5 * Math.PI - carro.orientacion);
 Recta recta1 = new Recta(tangentePos, -1, -tangentePos*pAx + pAy);

 double pBx = carro.x + 0.5 * carro.largo * Math.cos(carro.orientacion) - 0.5 * carro.ancho * Math.sin(carro.orientacion);
 double pBy = carro.y + 0.5 * carro.largo * Math.sin(carro.orientacion) + 0.5 * carro.ancho * Math.cos(carro.orientacion);
 double tangentePos2 = - Math.tan(0.5 * Math.PI - carro.orientacion - carro.volante);
 Recta recta2 = new Recta(tangentePos2, -1, -tangentePos2*pBx + pBy);
 Posicion centro = recta1.dimeCorte(recta2);
//System.out.println("PA:" + pAx + "," + pAy);
//System.out.println("PB:" + pBx + "," + pBy);
//System.out.println(centro.x + "," + centro.y);

 double angulo = centro.dimeAngulo(carro.dimePosicion());

//System.out.println("angulo:" + angulo);
 double radioX = centro.dimeDistancia(carro.dimePosicion()); 

double giros = - paso/radioX;

 if (carro.volante < 0 )
 {giros = - giros;
 }

 Posicion B = new Posicion(pBx, pBy);
 double anguloGirado = angulo + giros; 
//System.out.println("anguloGirado:" + anguloGirado);

 carro.x = centro.x + radioX *Math.cos(anguloGirado);
 carro.y = centro.y + radioX *Math.sin(anguloGirado);

/*  carro.orientacion = (anguloGirado + 0.5 * Math.PI) % (2 * Math.PI);*/
  carro.orientacion = (carro.orientacion + giros) % (2 * Math.PI);
for(int i=0; i<ultras.length; i++)
{ultras[i].orientacion = (ultras[i].orientacion + giros)%(2*Math.PI) ;
 ultras[i].x = (carro.dimePosicion().x + ( ultras[i].xRel * Math.cos(carro.orientacion)) - (ultras[i].yRel * Math.sin(carro.orientacion)));
 ultras[i].y = (carro.dimePosicion().y + (ultras[i].xRel * Math.sin(carro.orientacion)) + (ultras[i].yRel * Math.cos(carro.orientacion)));
}

}


calculoDistancias();
}
  }


int giro = 2;
double giroRad = giro * Math.PI / 180;

int incrementoVolante = 2;
double incVolanteRad = incrementoVolante * Math.PI / 180;


 class Girar implements ActionListener {
   public void actionPerformed(ActionEvent e) {

 carro.volante = carro.volante + incVolanteRad;

 carro.volante= (Double.parseDouble(String.valueOf(Math.round(carro.volante * 10000)))) / 10000;
System.out.println(carro.volante);

 graph.repaint();   
}
  }

class GirarMenos implements ActionListener {
   public void actionPerformed(ActionEvent e) {

 carro.volante = carro.volante - incVolanteRad;
 carro.volante= (Double.parseDouble(String.valueOf(Math.round(carro.volante * 10000)))) / 10000;
 System.out.println(carro.volante);
 graph.repaint();   

}
  }


/**
* lanzar un marco donde se pueden seleccionar entre
* qué iteraciones se desea realizar la ampliación
* de la gráfica
*/
 class OpcZoom implements ActionListener {
   public void actionPerformed(ActionEvent e) {
   }
  }

}  


