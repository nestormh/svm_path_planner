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

public class PlanoVisual extends Frame 
 implements WindowListener
 {
/* declaración del botón de cancelar */
/**
* botón de cancelar 
*/
public TextField[][] campos;

/**
* menú para la posibilidad de ampliar la gráfica
*/ 
Button actualizar = new Button ("Actualizar");

 DibujoVisual graph;
 int inicio = 0;
 Ultrasonido[] ultras; 
 Carro carro;
double coefA = 0.01732372;
double coefB = -1.7942;
 
 public PlanoVisual(Ultrasonido[] ultras, Carro carro){
  this.ultras=ultras;
  this.carro = carro;
campos = new TextField[ultras.length][4];   
for (int i=0; i<campos.length; i++)
{for (int j=0; j<2; j++)
{campos[i][j]= new TextField("255", 3);
}
campos[i][2] = new TextField(String.valueOf(256*Integer.parseInt(campos[i][0].getText()) + Integer.parseInt(campos[i][1].getText())), 5);
campos[i][3] = new TextField(String.valueOf(coefA*Double.parseDouble(campos[i][2].getText()) + coefB), 5);
}

/* declaración de un color, que se pone como fondo */
   Color t = new Color(210,210,255);
   setBackground(t);
   addWindowListener(this);
/* dibujo de la gráfica mediante la clase Dibujo, */
/* al que se le pasa como argumento el vector de */
/* error */
   graph=new DibujoVisual(ultras, carro);
/* título del marco */
   setTitle("Simulador de ultrasonidos");
/* añadir al marco el texto de area, el dibujo */
/* de la gráfica y el botón de cancelar */
 /*  add("North",texto1); */
   add(graph);
/*   add("East",cancelar);*/
/* imprimir el error de la primera iteración */   

 

/* declaración de un panel con los campos de texto */
/* y los botones */
  Panel q = new Panel();
  q.setLayout(new GridLayout(2 + campos.length, 1 + campos[0].length));

q.add(new Label("ID " ));
q.add(new Label("Byte alto" ));
q.add(new Label("Byte bajo" ));
q.add(new Label("cm" ));


for (int j=0; j<campos.length; j++)
{q.add(new Label(String.valueOf (j+1)));
 for (int k=0; k<campos[0].length; k++)
 {if( k!= 2)
  {q.add(campos[j][k]);}
  if( k==3)
  {campos[j][k].setEditable(false);}

 }

}
actualizar.addActionListener(new Actualizar());  
q.add(new Label(" "));

q.add(actualizar);
  add("East", q); /* añadir el panel al marco */


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



public void calculoDistancias()
{
   }

 class Actualizar implements ActionListener {
   public void actionPerformed(ActionEvent e) {
   for(int i=0; i<campos.length; i++)
   {
campos[i][2].setText(String.valueOf(256*Integer.parseInt(campos[i][0].getText()) + Integer.parseInt(campos[i][1].getText())));
double radio = coefA*Double.parseDouble(campos[i][2].getText()) + coefB;
campos[i][3].setText(String.valueOf(radio));
ultras[i].fijaRadio(Math.min(200,radio));
}
 
 graph.repaint();   
}
  }

  
  public void actualizar()
  {   for(int i=0; i<campos.length; i++)
   {
campos[i][2].setText(String.valueOf(256*Integer.parseInt(campos[i][0].getText()) + Integer.parseInt(campos[i][1].getText())));
double radio = coefA*Double.parseDouble(campos[i][2].getText()) + coefB;
campos[i][3].setText(String.valueOf(radio));
ultras[i].fijaRadio(Math.min(200,radio));
}
 graph.repaint();
  }

  public void actualizar(int i, int j, int k)
  {
for(int m=0; m<ultras.length; m++)
{int ident = ultras[m].dimeId();
 if (i==ident)
 {campos[m][0].setText(String.valueOf(j));
campos[m][1].setText(String.valueOf(k));
campos[m][2].setText(String.valueOf(256*Integer.parseInt(campos[m][0].getText()) + Integer.parseInt(campos[m][1].getText())));
double radio = coefA*Double.parseDouble(campos[m][2].getText()) + coefB;
campos[m][3].setText(String.valueOf(radio));
ultras[m].fijaRadio(Math.min(200,radio/2));
graph.repaint();

 }
}
  
  }
  
}  


