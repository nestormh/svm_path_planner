/**
 * Paquete que contiene las clases que permiten el control de los dispositivos
 * conectados a los puertos COM
 */
package carrito.server.serial;

import carrito.configura.Constantes;
import carrito.gps.CanvasRuta;

/**
 * Clase que permite controlar el vehículo a través de un puerto COM
 * @author Néstor Morales Hernández
 * @version 1.0
 */
public class ControlCarro /*implements Runnable*/ {
  private boolean acelera = false, gira = true;
  /** Punto central del volante del vehículo */
  public final static int CARRO_CENTRO = 5000;
  /** Ángulo de giro máximo del volante */
  public final static int CARRO_DIST = 4000;
  // Ángulo de giro máximo en radianes
  public final static double GIRO_MAX_RAD = Math.toRadians(30);
  // Velocidad máxima: 20 Km/h
  public final static double VEL_MAX = 255;
  public final static double VEL_MIN = 0;

    /** Objeto que hace de interfaz entre todas las variables comunes a la aplicación */
    private Constantes cte = null;
    /** Objeto SerialConnection que permite interactuar con el puerto COM */
    private SerialConnection puerto;

    /** Indica si la última vez se estaba acelerando o se estaba frenando */
    private int acelAnt = 0;
    /** Indica el sentido que llevaba la aceleración del vehículo en la última instrucción */
    private int posAnt = 0;
    private float giro;

    int errorAnt = 0;
    double derivativoAnt = 0;
    int comando = CARRO_CENTRO;
    double integral = 0;

    double errorAnt2 = 0;
    double derivativoAnt2 = 0;
    double comando2 = 0;
    double integral2 = 0;

    CanvasRuta canvas = null;

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
        reinit();
    }
    
    public ControlCarro(String nombrePuerto) {
        this.cte = cte;
        puerto = new SerialConnection(nombrePuerto);
        reinit();
    }
    
    public ControlCarro() {
      errorAnt = 0;
      derivativoAnt = 0;
      integral = 0;
      comando = CARRO_CENTRO;
      
      errorAnt2 = 0;
      derivativoAnt2 = 0;
      integral2 = 0;
      comando2 = 0;
      
    }
    
    /**
     * Inicializa el vehículo cada vez que un nuevo usuario toma el control del mismo
     */
    public void reinit() {
        System.out.println("Reinit");
        //puerto.ConsignaVolante = 32767;
        //puerto.setVolante(Constantes.CARRO_CENTRO);
        /*puerto.DesFrena(255);
        while (puerto.getDesfreno() == 0) {
            System.out.println("Inicializando:" + puerto.getDesfreno());
        }*/
        System.err.println("Ya se desfreno: " + puerto.getDesfreno());

        /*while (puerto.getIzq() != 1) {
            System.out.println("A la izq");
            puerto.setVolante(0);
        }*/
        //while (puerto.getVolante() < Constantes.CARRO_CENTRO)
        //puerto.setVolante(CARRO_CENTRO);

        acelAnt = 0;
        posAnt = puerto.getAvance();

        errorAnt = 0;
        derivativoAnt = 0;
        integral = 0;
        comando = CARRO_CENTRO;
        
        
        errorAnt2 = 0;
        derivativoAnt2 = 0;
        integral2 = 0;
        comando2 = 0;

        //activo = true;
        //Thread hilo = new Thread(this);
        //hilo.start();
    }

    /**
     * Indica el avance, retroceso o frenado al vehículo
     * @param aceleracion Aceleración. Si es negativa, se asume que el vehículo va
     * marcha atrás
     * @param frenado Indica la fuerza de frenado
     */
    /*public synchronized void setAvance(float aceleracion, float frenado) {
        if (frenado == 0) {
            if (puerto.getDesfreno() == 0) {
                System.out.println(System.currentTimeMillis() + "--Desfrenando: " + aceleracion + ", estado: " + puerto.getDesfreno());
                puerto.DesFrena(255);
                posAnt = puerto.getAvance();
                acelAnt = 0;
            } else {
                if (aceleracion >= 0) {
                    System.out.println(System.currentTimeMillis() + "--Acelerando: " + aceleracion);
                    puerto.Avanza((int)aceleracion);
                } else {
                    System.out.println(System.currentTimeMillis() + "--Retrocediendo: " + aceleracion);
                    puerto.Retrocede((int)aceleracion);
                }
            }
        } else {
            System.out.println(System.currentTimeMillis() + "--Frenando: " + frenado);
            puerto.Avanza(0);
            puerto.FrenaPasos((int)frenado);
            try {
                Thread.sleep(500);
            } catch (Exception e) {}
        }
    }*/
    public synchronized void setAvance(float aceleracion, float frenado) {
        System.out.println(System.currentTimeMillis() + "--Aceleracion: " + aceleracion);
        /*if (frenado == 0) {
            if (puerto.getDesfreno() == 0) {
                System.out.println(System.currentTimeMillis() + "--Desfrenando: " + aceleracion + ", estado: " + puerto.getDesfreno());
                puerto.DesFrena(255);
                posAnt = puerto.getAvance();
                acelAnt = 0;
            } else {
                if (aceleracion >= 0) { // Hacia delante
                    System.out.println(System.currentTimeMillis() + "--Acelerando: " + aceleracion);
                    if (acelAnt < 0) {
                        System.out.println("acelAnt < 0");
                        while (posAnt != puerto.getAvance()) { // Estaba andando hacia atrás
                            System.out.println("Estaba andando hacia detrás");
                            posAnt = puerto.getAvance();
                            puerto.Avanza(0);
                            puerto.Frena(255);
                            try {
                                Thread.sleep(100);
                            } catch (Exception e) {}
                        }
                    } else {
                        posAnt = puerto.getAvance();
                        acelAnt = (int) aceleracion;*/
                        puerto.Avanza((int) aceleracion);
                        /*
                    }
                } else {
                    System.out.println(System.currentTimeMillis() + "--Retrocediendo: " + aceleracion);
                    if (acelAnt > 0) {
                        System.out.println("acelAnt > 0");
                        while (posAnt != puerto.getAvance()) { // Estaba andando hacia delante
                            System.out.println("Estaba andando hacia delante");
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
                        puerto.Retrocede((int) - aceleracion);
                    }
                }
            }
        } else {
            System.out.println(System.currentTimeMillis() + "--Frenando: " + frenado);
            posAnt = puerto.getAvance();
            puerto.Avanza(0);
            //acelAnt = 0;
            puerto.FrenaPasos((int)frenado);
            try {
                Thread.sleep(500);
            } catch (Exception e) {}
            System.out.println(System.currentTimeMillis() + "--Estado: " + puerto.getDesfreno());
        }*/
    }

    /*public synchronized void setAvance(float aceleracion, float frenado) {
        this.aceleracion = (int)aceleracion;
        this.frenado = (int)frenado;
    }*/

    /*public void run() {
        while (activo) {
            int aceleracion = this.aceleracion;
            int frenado = this.frenado;
            if (frenado == 0) {
                while (puerto.getDesfreno() == 0) {
                    System.out.println("Desfrenando antes de continuar");
                    puerto.DesFrena(255);
                    posAnt = puerto.getAvance();
                }
                if (aceleracion >= 0) {    // Hacia delante
                    if (acelAnt < 0) {     // Estaba andando hacia atrás
                        System.out.println("Estaba andando hacia atrás");
                        while (posAnt != puerto.getAvance()) {
                            System.out.println("Deteniendo para cambiar el sentido de giro");
                            posAnt = puerto.getAvance();
                            puerto.FrenaPasos(1);
                            try {
                                Thread.sleep(500);
                            } catch (Exception e) {}
                        }
                    }
                    posAnt = puerto.getAvance();
                    puerto.Avanza(aceleracion);
                } else {
                    if (acelAnt >= 0) {     // Estaba andando hacia delante
                        System.out.println("Estaba andando hacia delante");
                        while (posAnt != puerto.getAvance()) {
                            System.out.println("Deteniendo para cambiar el sentido de giro");
                            posAnt = puerto.getAvance();
                            puerto.FrenaPasos(1);
                            try {
                                Thread.sleep(500);
                            } catch (Exception e) {}
                        }
                    }
                    posAnt = puerto.getAvance();
                    puerto.Retrocede(-aceleracion);
                }
                acelAnt = aceleracion;
            } else {
                System.out.println(System.currentTimeMillis() + "--Frenando: " + frenado);
                posAnt = puerto.getAvance();
                //acelAnt = 0;
                puerto.FrenaPasos(1);
                //try {
                //    Thread.sleep(10000);
                //} catch (Exception e) {}
                //System.out.println(System.currentTimeMillis() + "--Estado: " + puerto.getDesfreno());
            }
        }
    }*/

    /**
     * Indica el ángulo de giro del vehículo
     * @param ang Ángulo de giro
     */
    public void setGiro(float ang) {
        //System.out.println(System.currentTimeMillis() + "--Angulo antes: " + ang + ", actual: " + puerto.getVolante());
        if (ang < 0)
            ang = 0;
        if (ang > 65535)
            ang = 65535;
        //puerto.setVolante((int)ang);
        //System.out.println(System.currentTimeMillis() + "--Angulo despues: " + puerto.getVolante());
    }

    /**
     * Frena el vehículo en caso de que el cliente pierda el control del mismo
     */
    public void frenoEmergencia() {
        System.out.println("Frenado de emergencia. El cliente se ha desconectado");
        //activo = false;
        /*while (posAnt != puerto.getAvance()) { // Estaba andando hacia delante
            posAnt = puerto.getAvance();
            puerto.Avanza(0);
            puerto.Frena(255);
            try {
                Thread.sleep(100);
            } catch (Exception e) {}
        }*/
    }

    /**
     * Obtiene la aceleración anterior
     * @return Devuelve la aceleración anterior
     */
    public int getAcelAnt() {
        return acelAnt;
    }

    /**
     * Obtiene el valor del giro
     * @return Devuelve el valor del giro
     */
    public float getGiro() {
        return giro;
    }

    /**
     * Obtiene la posición del encoder
     * @return Devuelve la posición del encoder
     */
    public int getPosAnt() {
        return posAnt;
    }

    /**
     * Obtiene el objeto de comunicación con el puerto
     * @return Devuelve el objeto de comunicación con el puerto
     */
    public SerialConnection getPuerto() {
        return puerto;
    }

public void conduceSolo(double angulo, double velocidad, double kAlfa, double kDer, double kIntegral, 
        double kVel, double kAlfa2, double kDer2, double kIntegral2) {
  //System.out.println("**********");
  int sentido = 1;
  if (angulo > 0)
    sentido *= -1;
  System.out.println(sentido);
  int error = (int)(((CARRO_DIST * Math.abs(angulo)) / (Math.PI / 2)) * sentido);
  int alfadeseada = error + puerto.getVolante();

  double derivativo = kDer * (error - errorAnt) + kDer * derivativoAnt;
  // Dpos[0] = kdpos[0]*(error-errpos[0]) + kdpos[0]*Dpos[0];
  if (((comando > CARRO_CENTRO - CARRO_DIST) || (error > CARRO_CENTRO - CARRO_DIST))
      && ((comando < CARRO_CENTRO + CARRO_DIST) || (error < CARRO_CENTRO + CARRO_DIST)))
    integral = integral + kIntegral * errorAnt;
  comando = (int)(error * kAlfa + derivativo + integral) + CARRO_CENTRO;

  errorAnt = error;
  derivativoAnt = derivativo;

  if (comando > CARRO_DIST + CARRO_CENTRO)
    comando = CARRO_DIST + CARRO_CENTRO;
  if (comando < CARRO_CENTRO - CARRO_DIST)
    comando = CARRO_CENTRO - CARRO_DIST;

  /*System.out.println("*************************");
  System.out.println("Error: " + error);
  System.out.println("Derivativo: " + derivativo);
  System.out.println("Integral: " + integral);
  System.out.println("*************************");*/

  if (gira) {
    System.out.println("Girando: " + comando);
    puerto.setVolante(comando);
    System.out.println("puerto.setVolante(" + comando + ");");
    System.out.println(puerto.getVolante());    
    canvas.setAnguloRuedas(comando);
  }

  // Controlador PID para la velocidad
  double error2 = velocidad;
  double derivativo2 = kDer2 * (error2 - errorAnt2) + kDer2 * derivativoAnt2;
  if (((comando2 > VEL_MAX) || (error2 > VEL_MAX)) && 
          ((comando2 < VEL_MIN) || (error2 < VEL_MIN)))
      integral2 += kIntegral2 * errorAnt2;
  comando2 = (int)(error2 * kAlfa2 + derivativo2 + integral2 + kVel);

  errorAnt2 = error2;
  derivativoAnt2 = derivativo2;

  System.out.println("*************************");
  System.out.println("Giro: " + comando);
  System.out.println("KVel: " + kVel);
  System.out.println("KAlfa2: " + kAlfa2);
  System.out.println("Error2: " + error2);
  System.out.println("Derivativo2: " + derivativo2);
  System.out.println("Integral2: " + integral2);

  if (acelera) {      
      System.out.println("Acelerando");
      if (comando2 >= 0) {
        puerto.Avanza( (int) comando2);
        System.out.println("puerto.Avanza((int)" + comando2 + ");");    
      } else {
          System.out.println("El comando es negativo");
      }
  }
  System.out.println("*************************");
}

  public void simulaConduceSolo(double angulo, double velocidad, double kAlfa, double kDer, double kIntegral,
        double kVel, double kAlfa2, double kDer2, double kIntegral2) {  
  int sentido = 1;
  if (angulo > 0)
    sentido *= -1;
  System.out.println(sentido);
  int error = (int)(((CARRO_DIST * Math.abs(angulo)) / (Math.PI / 2)) * sentido);
  //int alfadeseada = error + CARRO_CENTRO;

  double derivativo = kDer * (error - errorAnt) + kDer * derivativoAnt;
  // Dpos[0] = kdpos[0]*(error-errpos[0]) + kdpos[0]*Dpos[0];
  if (((comando > CARRO_CENTRO - CARRO_DIST) || (error > CARRO_CENTRO - CARRO_DIST))
      && ((comando < CARRO_CENTRO + CARRO_DIST) || (error < CARRO_CENTRO + CARRO_DIST)))
    integral = integral + kIntegral * errorAnt;
  comando = (int)(error * kAlfa + derivativo + integral) + CARRO_CENTRO; 

  errorAnt = error;
  derivativoAnt = derivativo;

  if (comando > CARRO_DIST + CARRO_CENTRO)
    comando = CARRO_DIST + CARRO_CENTRO;
  if (comando < CARRO_CENTRO - CARRO_DIST)
    comando = CARRO_CENTRO - CARRO_DIST;

  /*System.out.println("*************************");
  System.out.println("Error: " + error);
  System.out.println("Derivativo: " + derivativo);
  System.out.println("Integral: " + integral);
  System.out.println("*************************");*/

  if (gira) {
      System.out.println("Simulando");
    System.out.println("puerto.setVolante(" + comando + ");");
    canvas.setAnguloRuedas(comando);
  }

  // Controlador PID para la velocidad
  double error2 = velocidad;
  double derivativo2 = kDer2 * (error2 - errorAnt2) + kDer2 * derivativoAnt2;
  if (((comando2 > VEL_MAX) || (error2 > VEL_MAX)) && 
          ((comando2 < VEL_MIN) || (error2 < VEL_MIN)))
      integral2 += kIntegral2 * errorAnt2;
  comando2 = (int)(error2 * kAlfa2 + derivativo2 + integral2 + kVel);

  errorAnt2 = error2;
  derivativoAnt2 = derivativo2;

  System.out.println("*************************");
  System.out.println("Dif. Velocidad: " + velocidad);
  System.out.println("Error2: " + error2);
  System.out.println("Derivativo2: " + derivativo2);
  System.out.println("Integral2: " + integral2);
  System.out.println("Integral2: " + kVel);

  if (acelera) {
    System.out.println("puerto.Avanza((int)" + comando2 + ");");
    System.out.println("*************************");
  }
}

public boolean isAcelera() {
return acelera;
}

public boolean isGira() {
return gira;
}

public void setAcelera(boolean acelera) {
this.acelera = acelera;
}

public void setGira(boolean gira) {
this.gira = gira;
}

public void setCanvas(CanvasRuta canvas) {
this.canvas = canvas;
}

}
