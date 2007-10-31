package carrito.server.serial;
/*
 * Guistub.java
 *
 * Created on 19 de junio de 2006, 9:56
 */

/**
 *
 * @author  jonay
 */


import java.io.*;
import java.awt.TextArea;
import java.awt.*;
import java.awt.event.*;



public class Guistub extends Frame implements KeyListener {
    int vel = 0;
    SerialConnection puerto;
    static String PuertoAbrir;
    /** Creates a new instance of Guistub */
    public Guistub() {
    }


     public Guistub(String Puerto) {
        super("Guistub Consola");

        setLayout(null);
        addKeyListener(this);
	puerto = new SerialConnection(Puerto,9600,0,0,8,1,0);
        while (puerto.getVolante() > 0)
                puerto.setVolante(0);

       puerto.ConsignaVolante = 0;
    }


     public Guistub(String s, String Puerto) {
        super(s);

        setLayout(null);
        addKeyListener(this);
	puerto = new SerialConnection(Puerto,9600,0,0,8,1,0);
        while (puerto.getVolante() > 0)
                puerto.setVolante(0);
        puerto.ConsignaVolante = 0;

    }


    public static void main(String args[]) {


        if (args.length > 0)
            PuertoAbrir = args[0];
        else
            PuertoAbrir = "COM1";
       Guistub Guistub = new Guistub(PuertoAbrir);

       Guistub.setSize(400,400);
       Guistub.setVisible(true);




  }


    public void keyPressed(KeyEvent evt) {
          // Called when the user has pressed a key, which can be
          // a special key such as an arrow key.  If the key pressed
          // was one of the arrow keys, move the square (but make sure
          // that it doesn't move off the edge, allowing for a
          // 3-pixel border all around the applet).  SQUARE_SIZE is
          // a named constant that specifies the size of the square.

      char a = 0;
      int key = evt.getKeyCode();  // Keyboard code for the pressed key.


        if (key == 87) { //W avance
              if ((vel < 0) && (vel + 5 > 0)) {
                puerto.DesFrena(255);
                return;
              }
              vel = vel + 5;

              if (vel > 255)
                  vel = 255;
              if (vel >= 0) {
                  puerto.Avanza(vel);

              } else if (vel < 0)
                  puerto.DesFrenaPasos(100);
              System.out.println("Velocidad " + vel);
          }

          if (key == 83) { // S retroceso
              vel = vel - 5;
              if (vel < -255)
                  vel = -255;
              if (vel >= 0) {
                  puerto.Avanza(vel);

              } else if (vel < 0)
                  puerto.FrenaPasos(100);

              System.out.println("Velocidad " + vel);

          }

          if (key == 65) { //A Izquierda
              System.out.println("Angulo " + puerto.getVolante());
              if (puerto.getVolante() > 30)
                puerto.setRVolante(-30);

          }

          if (key == 68) { // D derecha
              System.out.println("Angulo " + puerto.getVolante());
              puerto.setRVolante(30);
          }




   }

    public void keyReleased(KeyEvent e) {

    }

    public void keyTyped(KeyEvent e) {

    }

  // end keyPressed()

    public SerialConnection getGuistub() {
        return puerto;
    }

}
