/**
 * Paquete que contiene todas las clases correspondientes a la aplicación servidor
 */
package carrito.server;

import java.awt.*;
import java.rmi.*;
import java.rmi.server.*;
import java.util.*;

import carrito.configura.*;
import carrito.server.serial.*;

/**
 * Objeto que va a recibir las instrucciones remotas desde el cliente y va a
 * traducir estas llamadas a cada uno de los objetos encargados de cada tarea
 * @author Néstor Morales Hernández
 * @version 1.0
 */
public class ServidorRMI extends UnicastRemoteObject implements InterfazRMI, Runnable {
    /** Objeto que hace de interfaz entre todas las variables comunes a la aplicación */
    private Constantes cte = null;
    /** Objeto de control del vehículo */
    private ControlCarro control = null;
    /** Objeto de control del motor de las cámaras */
    private ControlCamara camara = null;
    /** Objeto que controla el zoom de las cámaras */
    private ControlZoom CZoom = null;
    /** Objeto que permite la creación de números aleatorios */
    private Random rnd = null;
    /** Identificador del dueño del control del vehículo */
    private int joyOwner = Constantes.NULLINT;
    /** Indica si el dueño del control del vehículo está activo */
    private boolean activo = false;
    /** Indica si el vehículo está funcionando */
    private boolean funcionando = false;
    /** Hilo de ejecución para comprobar si el cliente que tiene el control se desconecta */
    private Thread hilo = null;

    /**
     * Constructor. Inicializa las variables y crea el hilo de ejecución.
     * @param cte Objeto que hace de interfaz entre todas las variables comunes a la aplicación
     * @param control Objeto de control del vehículo
     * @param camara Objeto de control del motor de las cámaras
     * @param CZoom Objeto que controla el zoom de las cámaras
     * @throws RemoteException
     */
    public ServidorRMI(Constantes cte, ControlCarro control, ControlCamara camara, ControlZoom CZoom) throws RemoteException {
        this.cte = cte;
        this.control = control;
        this.camara = camara;
        this.CZoom = CZoom;
        this.rnd = new Random(System.currentTimeMillis());

        hilo = new Thread(this);
        hilo.start();
    }

    /**
     * Comprueba si el cliente se desconectó, y en caso afirmativo, frena el
     * vehículo
     */
    private synchronized void inactivos() {
        if ((! activo) && (funcionando)) {
            control.frenoEmergencia();
            joyOwner = Constantes.NULLINT;
            funcionando = false;
        }
        // Pone activo a false, lo cual significa que el cliente tiene 1 segundo
        // para volverlo a poner a true, para indicar que no se ha desconectado
        activo = false;
    }

    /**
     * Hilo de ejecución que hace sucesivas llamadas a la función <i>inactivos</i>
     */
    public void run() {
        while(true) {
            inactivos();
            try {
                Thread.sleep(2000);
            } catch (Exception e) {}
        }
    }

    /**
     * Devuelve una lista de objetos VideoCltConfig, los cuales describen un cliente
     * multimedia que va a corresponder a cada uno de los servidores multimedia en la
     * aplicación servidor
     * @return Devuelve una lista de objetos VideoCltConfig
     * @throws RemoteException
     */
    public VideoCltConfig[] getVideos() throws RemoteException {
        VideoCltConfig[] retorno = new VideoCltConfig[cte.getNumEmisores()];
        for (int i = 0; i < retorno.length; i++) {
            retorno[i] = new VideoCltConfig("", 1, null, 10, null, Constantes.NULLINT,
                                            Constantes.NULLFLOAT, null, null);
            retorno[i].setIp(cte.getEmisor(i).getIp());
            retorno[i].setPort(cte.getEmisor(i).getPort());
        }
        return retorno;
    }

    /**
         * Obtiene el tamaño del vídeo transmitido desde el dispositivo indicado
         * @param cam Número de dispositivo al que se hace referencia
         * @return Devuelve el tamaño de vídeo solicitado
         * @throws RemoteException
     */
    public Dimension getSize(int cam) {
        return new Dimension(cte.getEmisor(cam).getWidth(), cte.getEmisor(cam).getHeight());
    }

    /**
     * Comprueba que el cliente que mandó un comando de control del vehículo esté
     * activo como dueño de dicho control. Además, si <i>funcionando</i> está a <i>false</i>,
     * lo cambia a <i>true</i> para indicar que ya está andando
     * @param own Identificador del dueño del control del vehículo
     * @return Devuelve si está activo o no
     */
    private synchronized boolean isActivo(int own) {
        funcionando = true;
        if (own == joyOwner) {
            activo = true;
            return true;
        } else {
            return false;
        }
    }

    /**
         * Indica nuevos parámetros de avance al vehículo, para llevar a cabo su control
         * @param own Identificador de dueño del control, para asegurarnos que nadie intercepta
         * el envío de comandos
         * @param aceleracion Indica la aceleración que se le va a mandar al vehículo
         * @param frenado Indica la fuerza de frenado
         * @param giro Indica el ángulo de giro del vehículo
         * @return Devuelve <i>true</i> si todo ha ido bien. Si no, es que el usuario
         * ha perdido el control del vehículo
         * @throws RemoteException
     */
    public boolean avanzaCarro(int own, float aceleracion, float frenado, float giro) throws RemoteException {
        if (! isActivo(own))
            return false;
        if (aceleracion != Constantes.NULLINT)
            control.setAvance(aceleracion, frenado);
        control.setGiro(giro);
        return true;
    }

    /**
     * Indica el ángulo lateral de las cámaras
     * @param id Cámara a la que se hace referencia
     * @param angulo Ángulo indicado
     * @throws RemoteException
     */
    public void setAnguloCamaras(int id, int angulo) throws RemoteException {
        camara.setAngulo(id, angulo);
    }

    /**
     * Indica el ángulo en altura de las cámaras
     * @param angulo Ángulo indicado
     * @throws RemoteException
     */
    public void setAlturaCamaras(int angulo) throws RemoteException {
        camara.setAltura(angulo);
    }

    /**
     * Indica el zoom de una determinada cámara
     * @param zoom Indica el nuevo zoom
     * @param id Cámara a la que se hace referencia
     * @throws RemoteException
     */
    public void setZoom(int zoom, int id) throws RemoteException {
        CZoom.setZoom(zoom, id);
    }

    /**
     * Solicita el control del vehículo
     * @return Devuelve -1 si se deniega el control, u otro número en caso de que
     * este haya sido concedido, indicando el identificador de posesión del control
     * @throws RemoteException
     */
    public synchronized int getJoystick() throws RemoteException {
        // Si está siendo usado, devuelve -1
        if (joyOwner != Constantes.NULLINT) {            
            return Constantes.NULLINT;
            // Si no, reinicia las variables de control y le da un nuevo identificador
            // al cliente
        } else {
            activo = true;
            funcionando = false;
            joyOwner = rnd.nextInt();
            control.reinit();
            return joyOwner;
        }
    }

    /**
     * Libera el control del vehículo
     * @param id Identificador de posesión del vehículo
     * @throws RemoteException
     */
    public synchronized void freeJoystick(int id) throws RemoteException {
        if (joyOwner != id)
            return;
        joyOwner = Constantes.NULLINT;
    }
    
    public boolean frenoTotal(int id) throws RemoteException {
        
        if (! isActivo(id))
            return false;
        
        control.frenadoTotal();
        return true;        
    }
    
    public boolean desfrenoTotal(int id) throws RemoteException {
        
        if (! isActivo(id))
            return false;

        control.desfrenadoTotal();
        return true;        
    }
    
    public boolean resetAvance(int id, boolean retrocede) throws RemoteException {
        if (! isActivo(id))
            return false;

        control.resetAvance(retrocede);
        return true;
    }
}
