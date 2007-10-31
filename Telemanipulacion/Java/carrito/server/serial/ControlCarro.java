/**
 * Paquete que contiene las clases que permiten el control de los dispositivos
 * conectados a los puertos COM
 */
package carrito.server.serial;

import carrito.configura.*;

/**
 * Clase que permite controlar el vehículo a través de un puerto COM
 * @author Néstor Morales Hernández
 * @version 1.0
 */
public class ControlCarro /*implements Runnable*/ {
    private final int FRENO = 255;
    
    boolean frenado = false;
    /** Objeto que hace de interfaz entre todas las variables comunes a la aplicación */
    private Constantes cte = null;
    /** Objeto SerialConnection que permite interactuar con el puerto COM */
    private SerialConnection puerto;

    /** Indica si la última vez se estaba acelerando o se estaba frenando */
    private int acelAnt = 0;
    /** Indica el sentido que llevaba la aceleración del vehículo en la última instrucción */
    private int posAnt = 0;

    /*private static int aceleracion = 0;
    private static int frenado = 0;
    private static boolean activo = false;*/

    /**
     * Constructor. Abre la conexión con el puerto COM e inicializa el vehículo
     * @param cte Constantes
     */
    public ControlCarro(Constantes cte) {
        this.cte = cte;
        puerto = new SerialConnection(cte.getCOMCarrito(), 9600,0,0,8,1,0);

        //reinit();
    }

    /**
     * Inicializa el vehículo cada vez que un nuevo usuario toma el control del mismo
     */
    public void reinit() {
        System.out.println("Reinit");
        if (cte.isCanInitDesfreno()) {
          System.out.println("Se va a proceder a desfrenar el vehículo");
          puerto.DesFrena(255);
          while (puerto.getDesfreno() == 0) {
            puerto.DesFrena(255);
          }
          System.err.println("Vehículo desfrenado");
        }

        if (cte.isCanInitVolante()) {
          System.out.println("Se va a proceder a calibrar las ruedas");
          while (puerto.getIzq() == 0)
            puerto.setVolante(0);
          while (puerto.getVolante() < Constantes.CARRO_CENTRO) {
              System.out.println(puerto.getVolante());
            puerto.setVolante(Constantes.CARRO_CENTRO);
          }
          System.out.println("Ruedas calibradas");
        }

        acelAnt = 0;
        posAnt = puerto.getAvance();
    }

    /**
     * Indica el avance, retroceso o frenado al vehículo
     * @param aceleracion Aceleración. Si es negativa, se asume que el vehículo va
     * marcha atrás
     * @param frenado Indica la fuerza de frenado
     */
    public synchronized void setAvance(float aceleracion, float frenado) {
        System.out.println(System.currentTimeMillis() + "--Aceleracion: " + aceleracion +
                ", frenado: " + frenado);
        if (frenado == 0) {
            if ((puerto.getDesfreno() == 0) && (cte.isCanDesfreno())) {
                System.out.println(System.currentTimeMillis() + "--Desfrenando: " + puerto.getDesfreno());
                puerto.DesFrena(255);
                posAnt = puerto.getAvance();
                acelAnt = 0;
            } else {
                if ((aceleracion >= 0) && (cte.isCanAvance())) { // Hacia delante
                    //System.out.println(System.currentTimeMillis() + "--Acelerando: " + aceleracion);
                    // Estaba andando hacia atrás. Debemos frenar el vehículo antes de continuar
                    if ((acelAnt < 0) && (cte.isCanFreno())) {
                        System.out.println("acelAnt < 0");
                        System.out.println("Estaba andando hacia detrás");
                        while (Math.abs(posAnt - puerto.getAvance()) > 10) {
                          System.out.println("Frenando");
                            posAnt = puerto.getAvance();
                            puerto.Avanza(0);
                            puerto.Frena(255);
                            try {
                                Thread.sleep(100);
                            } catch (Exception e) {}
                        }
                    } else {                    
                        posAnt = puerto.getAvance();
                        acelAnt = (int) aceleracion;
                        puerto.Avanza((int) aceleracion);
                    }
                } else if (cte.isCanRetroceso()) {
                    //System.out.println(System.currentTimeMillis() + "--Retrocediendo: " + aceleracion);
                    if ((acelAnt > 0) && (cte.isCanFreno())) {
                        System.out.println("acelAnt > 0");
                        System.out.println("Estaba andando hacia delante");
                        while (Math.abs(posAnt - puerto.getAvance()) > 10) {
                          System.out.println("Frenando");
                            posAnt = puerto.getAvance();
                            puerto.Avanza(0);
                            puerto.Frena(255);
                            try {
                                Thread.sleep(100);
                            } catch (Exception e) {}
                        }
                    } else {                    
                        //System.out.println("Retrocediendo: " + (-aceleracion));
                        posAnt = puerto.getAvance();
                        acelAnt = (int) aceleracion;
                        puerto.Retrocede((int) - aceleracion);
                    }
                }
            }
        } else if (cte.isCanFreno()) {
            System.out.println(System.currentTimeMillis() + "--Frenando: " + frenado);
            posAnt = puerto.getAvance();
            puerto.Avanza(0);
            puerto.FrenaPasos((int)frenado);
            try {
                Thread.sleep(500);
            } catch (Exception e) {}
            System.out.println(System.currentTimeMillis() + "--Estado: " + puerto.getDesfreno());
        }
    }

    /**
     * Indica el ángulo de giro del vehículo
     * @param ang Ángulo de giro
     */
    public void setGiro(float ang) {
      if (cte.isCanGiro()) {
        if (ang < 0)
          ang = 0;
        if (ang > 65535)
          ang = 65535;
        puerto.setVolante( (int) ang);
      }
    }

    /**
     * Frena el vehículo en caso de que el cliente pierda el control del mismo
     */
    public void frenoEmergencia() {
      puerto.Avanza(0);
      if (cte.isCanFrenoEmergencia()) {
        System.out.println("Frenado de emergencia. El cliente se ha desconectado");
        while (Math.abs(posAnt - puerto.getAvance()) > 10) { // Estaba andando hacia delante
            posAnt = puerto.getAvance();
            puerto.Avanza(0);
            puerto.Frena(255);
            try {
                Thread.sleep(100);
            } catch (Exception e) {}
        }
      }
    }
    
    public void frenadoTotal() {        
        if (frenado) return;
        
        System.out.println("Frenando totalmente");
        
        puerto.Avanza(0);
        puerto.FrenaTotal();
                                
        frenado = true;
    }
    
    public void desfrenadoTotal() {
        System.out.println("Desfreno total");
        
        puerto.DesFrenaTotal();
        
        frenado = false;
    }
    
    public void resetAvance(boolean retrocede) {
        
        if (! retrocede) {
            System.out.println("Avanza 0");        
            puerto.Avanza(0);
        } else {
            System.out.println("Retrocede 0");
            puerto.Retrocede(0);
        }
    }
}
