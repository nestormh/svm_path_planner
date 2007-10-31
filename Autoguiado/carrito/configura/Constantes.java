/**
 * Paquete que contiene todas las clases descriptoras de las propiedades que
 * afectan a la aplicación
 */
package carrito.configura;

import java.io.*;
import java.rmi.*;
import java.util.*;

import java.awt.*;
import javax.swing.*;

import carrito.media.*;
import carrito.server.*;

/**
 * Clase que hace de interfaz para que todos los objetos de la aplicación puedan
 * acceder a aquellas variables que son comunes a toda la aplicación
 * @author Néstor Morales Hernández
 * @version 1.0
 */
public class Constantes {
    // Constantes
    /** Valor nulo para enteros */
    public final static int NULLINT = -1;
    /** Valor nulo para punto flotante */
    public final static float NULLFLOAT = -10.0f;
    /** Operación desactivada */
    public final static int DESACTIVADA = -555;

    /** Punto central del volante del vehículo */
    public final static int CARRO_CENTRO = 32767;
    /** Ángulo de giro máximo del volante */
    public final static int CARRO_DIST = 4400;

    // Variables de configuración
    /** Array con los descriptores de los emisores de streaming */
    private VideoSvrConf emisores[] = null;
    /** Array con los descriptores de los emisores de streaming */
    private VideoCltConfig receptores[] = null;
    /** Identificadores de las instancias de vídeo */
    private int idVideos[] = null;
    /** Dirección IP del servidor */
    private String ip = "127.0.0.1";
    /** Valor de caché de cliente */
    private int cltCaching = 300;
    // Valores extremos del Joystick
    private float maxDerecha = NULLFLOAT, maxIzquierda = NULLFLOAT, maxArriba = NULLFLOAT, maxAbajo = NULLFLOAT;
    private float XNone = NULLFLOAT, XDif= NULLFLOAT, YNone = NULLFLOAT, YDif= NULLFLOAT;
    /** Puerto de serie para controlar el ángulo de la cámara */
    private String COMCamara = "COM9";
    /** Puerto de serie para controlar el vehículo */
    private String COMCarrito = "COM10";

    /**
     * Constructor
     */
    public Constantes() {}

    /**
     * Método que automatiza la emisión de los mensajes de error en la aplicación
     * @param msg Mensaje de error
     * @param e Excepción que originó el mensaje
     */
    public static void mensaje(String msg, Exception e) {
        JOptionPane.showMessageDialog(null, msg + e.getMessage(), "Error", JOptionPane.ERROR_MESSAGE);
    }

    /**
     * Método que automatiza la emisión de los mensajes de error en la aplicación
     * @param msg Mensaje de error
     */
    public static void mensaje(String msg) {
        JOptionPane.showMessageDialog(null, msg, "Error", JOptionPane.ERROR_MESSAGE);
    }

    /**
     * Método que abre el fichero de configuración del servidor e inicializa todas
     * las variables de la aplicación servidor
     */
    public void openServer() {
        // Obtiene la lista de dispositivos conectados actualmente
        String dispName[] = Media.listaDispositivos();
        // Crea un array vacío de emisores
        emisores = new VideoSvrConf[dispName.length];

        // Crea los emisores del array con los valores por defecto
        for (int i = 0; i < emisores.length; i++) {
            emisores[i] = new VideoSvrConf(dispName[i], i);
        }

        // Abre el fichero "configsrv.dat", si no existe lo crea por defecto
        File conf = new File("configsrv.dat");
        while (! conf.exists()) {
            saveServer();
        }

        // Variables InputStream
        FileInputStream fis = null;
        ObjectInputStream ois = null;

        try {
            fis = new FileInputStream(conf);
            ois = new ObjectInputStream(fis);

            // Lee los objetos del tipo VideoSvrConf
            int tamano = ois.readInt();
            emisores = new VideoSvrConf[tamano];
            for (int i = 0; i < tamano; i++) {
                emisores[i] = (VideoSvrConf)ois.readObject();
            }

            // Puertos COM
            COMCarrito = ois.readUTF();
            COMCamara = ois.readUTF();

            // Cierra el Stream
            ois.close();
            fis.close();
        } catch(FileNotFoundException fnfe) {
            System.out.println("Error al acceder al fichero de configuración. Saliendo");
            System.out.println(fnfe.getMessage());
            System.exit(0);
        }catch(ClassNotFoundException fnfe) {
            System.out.println("Error al acceder al fichero de configuración. Saliendo");
            System.out.println(fnfe.getMessage());
            System.exit(0);
        } catch(IOException ioe) {
            System.out.println("Error al acceder al fichero de configuración. Saliendo");
            System.out.println(ioe.getMessage());
            System.exit(0);
        }
    }

    /**
     * Método que guarda en el fichero de configuración las variables de configuración
     * de la aplicación. Si no existe, se crea.
     */
    public void saveServer() {
        // Abre el fichero y si no existe lo crea
        File conf = new File("configsrv.dat");
        if (! conf.exists()) {
            System.out.println("No existe el fichero de configuración. Se creará uno");
            try {
                if (!conf.createNewFile()) {
                    System.out.println("No se pudo crear el fichero de configuración. Saliendo");
                    System.exit(0);
                } else {
                    System.out.println("El fichero se creó con éxito");
                }
            } catch(IOException ioe) {
                System.out.println("No se pudo crear el fichero de configuración. Saliendo");
                System.out.println(ioe.getMessage());
                System.exit(0);
            }
        }

        // OutputStreams
        FileOutputStream fos = null;
        ObjectOutputStream oos = null;
        try {
            fos = new FileOutputStream(conf);
            oos = new ObjectOutputStream(fos);

            // Escribe los objetos del tipo VideoSvrConf
            oos.writeInt(emisores.length);
            for (int i = 0; i < emisores.length; i++) {
                oos.writeObject(emisores[i]);
            }
            oos.writeUTF(COMCarrito);
            oos.writeUTF(COMCamara);

            // Cierra los OutputStream
            oos.close();
            fos.close();
        } catch(FileNotFoundException fnfe) {
            System.out.println("Error al acceder al fichero de configuración. Saliendo");
            System.out.println(fnfe.getMessage());
            System.exit(0);
        } catch(IOException ioe) {
            System.out.println("Error al acceder al fichero de configuración. Saliendo");
            System.out.println(ioe.getMessage());
            System.exit(0);
        }
    }

    /**
     * Método que abre el fichero de configuración del cliente e inicializa todas
     * las variables de la aplicación cliente
     */

    public void openCliente() {
        // Abre el fichero "configclt.dat", y si no existe crea uno por defecto
        File conf = new File("configclt.dat");
        while (! conf.exists()) {
            saveCliente();
        }

        // InputStreams
        FileInputStream fis = null;
        ObjectInputStream ois = null;

        try {
            fis = new FileInputStream(conf);
            ois = new ObjectInputStream(fis);

            // Lee las variables
            ip = ois.readUTF();
            cltCaching = ois.readInt();
            maxIzquierda = ois.readFloat();
            maxDerecha = ois.readFloat();
            maxArriba = ois.readFloat();
            maxAbajo = ois.readFloat();
            XNone = ois.readFloat();
            XDif = ois.readFloat();
            YNone = ois.readFloat();
            YDif = ois.readFloat();

            // Cierra los InputStream
            ois.close();
            fis.close();
        } catch(FileNotFoundException fnfe) {
            System.out.println("Error al acceder al fichero de configuración. Saliendo");
            System.out.println(fnfe.getMessage());
            System.exit(0);
        } catch(IOException ioe) {
            System.out.println("Error al acceder al fichero de configuración. Saliendo");
            System.out.println(ioe.getMessage());
            System.exit(0);
        }
    }

    /**
     * Método que guarda en el fichero de configuración las variables de configuración
     * de la aplicación. Si no existe, se crea uno.
     */
    public void saveCliente() {
        // Abre el fichero de configuración, y si no existe crea uno
        File conf = new File("configclt.dat");
        if (! conf.exists()) {
            System.out.println("No existe el fichero de configuración. Se creará uno");
            try {
                if (!conf.createNewFile()) {
                    System.out.println("No se pudo crear el fichero de configuración. Saliendo");
                    System.exit(0);
                } else {
                    System.out.println("El fichero se creó con éxito");
                }
            } catch(IOException ioe) {
                System.out.println("No se pudo crear el fichero de configuración. Saliendo");
                System.out.println(ioe.getMessage());
                System.exit(0);
            }
        }

        // OutputStreams
        FileOutputStream fos = null;
        ObjectOutputStream oos = null;
        try {
            fos = new FileOutputStream(conf);
            oos = new ObjectOutputStream(fos);

            // Escribe las variables
            oos.writeUTF(ip);
            oos.writeInt(cltCaching);
            oos.writeFloat(maxIzquierda);
            oos.writeFloat(maxDerecha);
            oos.writeFloat(maxArriba);
            oos.writeFloat(maxAbajo);
            oos.writeFloat(XNone);
            oos.writeFloat(XDif);
            oos.writeFloat(YNone);
            oos.writeFloat(YDif);

            // Cierra los OutputStream
            oos.close();
            fos.close();
        } catch(FileNotFoundException fnfe) {
            System.out.println("Error al acceder al fichero de configuración. Saliendo");
            System.out.println(fnfe.getMessage());
            System.exit(0);
        } catch(IOException ioe) {
            System.out.println("Error al acceder al fichero de configuración. Saliendo");
            System.out.println(ioe.getMessage());
            System.exit(0);
        }
    }

    /**
     * Obtiene el objeto descriptor del emisor indicado
     * @param i Número de emisor que se desea conocer
     * @return El descriptor del emisor indicado
     */
    public VideoSvrConf getEmisor(int i) {
        return emisores[i];
    }

    /**
     * Establece el nuevo array de emisores
     * @param emisores Nuevo array con el descriptor de los emisores
     */
    public void setEmisores(VideoSvrConf[] emisores) {
        this.emisores = emisores;
    }

    /**
     * Obtiene el número de emisores actual
     * @return Devuelve el número actual de emisores
     */
    public int getNumEmisores() {        
        return emisores.length;
    }

    /**
     * Obtiene el descriptor para el receptor indicado
     * @param i Número del descriptor que se desea conocer
     * @return el descriptor para el receptor indicado
     */
    public VideoCltConfig getReceptor(int i) {
        return receptores[i];
    }

    /**
     * Establece la configuración de todos los receptores de la aplicación en base
     * a la configuración actual de los emisores de la aplicación servidor
     * @param rmi Objeto RMI
     */
    public void setReceptores(InterfazRMI rmi) {
        int posX = 320;
        int posY = 75;
        int maxHeight = 0;
        VideoCltConfig[] receptores = null;

        // Obtiene la configuración de los emisores en la aplicación servidor
        try {
            receptores = rmi.getVideos();
        } catch(RemoteException re) {
            mensaje("Error al obtener la configuración de los vídeos. ", re);
        }

        // Establece la posición de los receptores en base al tamaño de los vídeos
        for (int i = 0; i < receptores.length; i++) {
            Dimension dim = null;
            // Obtiene el tamaño del vídeo i (Si hay una excepción, se usa uno por defecto)
            try {
                dim = rmi.getSize(i);
            } catch(RemoteException re) {
                dim = new Dimension(176, 144);
            }
            // Si la posición X se sale de la pantalla, se pone a la izquierda y debajo de la
            // línea de vídeos anterior. Se reinicia la obtención del mayor tamaño de vídeo
            if (posX + dim.width > Toolkit.getDefaultToolkit().getScreenSize().width - 350) {
                posX = 320;
                posY += maxHeight + 50;
                maxHeight = 0;
            }
            // Si se sale por la pantalla de debajo, se vuelve a empezar por arriba
            if (posY + dim.height > Toolkit.getDefaultToolkit().getScreenSize().height) {
                Random r = new Random(System.currentTimeMillis());
                posY = r.nextInt(20);
            }
            // Actualiza el mayor tamaño de vídeo (altura)
            if (dim.height > maxHeight)
                maxHeight = dim.height;
            // Establece la posición del video
            receptores[i].setPosicion(new Dimension(posX, posY));
            // Actualiza la próxima posición X
            posX += dim.width + 30;
        }

        // Actualiza el valor del array descriptor de los receptores
        this.receptores = receptores;
    }

    /**
     * Obtiene el número de receptores actual
     * @return Devuelve el número de receptores actual
     */
    public int getNumReceptores() {
        return receptores.length;
    }

    /**
     * Establece el vector con los identificadores de las instancias VLC
     * @param idVideos Vector con los identificadores de las instancias VLC
     */
    public void setIdVideos(int[] idVideos) {
        this.idVideos = idVideos;
    }

    /**
     * Obtiene el identificador de la instancia VLC correspondiente al receptor i
     * @param i Número de receptor
     * @return Identificador de la instancia VLC
     */
    public int getIdVideo(int i) {
        return idVideos[i];
    }

    /**
     * Obtiene el número de instancias VLC
     * @return Devuelve el número de instancias VLC
     */
    public int getNumVideos() {
        return idVideos.length;
    }

    /**
     * Establece el valor de la IP del servidor
     * @param ip Valor de la IP del servidor
     */
    public void setIp(String ip) {
        this.ip = ip;
    }

    /**
     * Establece el máximo valor de frenado del Joystick
     * @param maxAbajo Máximo valor de frenado del Joystick
     */
    public void setMaxAbajo(float maxAbajo) {
        this.maxAbajo = maxAbajo;
    }

    /**
     * Establece el máximo valor de aceleración del Joystick
     * @param maxArriba Máximo valor de aceleración del Joystick
     */
    public void setMaxArriba(float maxArriba) {
        this.maxArriba = maxArriba;
    }

    /**
     * Establece el máximo valor de giro hacia la derecha del Joystick
     * @param maxDerecha Máximo valor de giro hacia la derecha del Joystick
     */
    public void setMaxDerecha(float maxDerecha) {
        this.maxDerecha = maxDerecha;
    }

    /**
     * Establece el máximo valor de giro hacia la izquierda del Joystick
     * @param maxIzquierda Máximo valor de giro hacia la izquierda del Joystick
     */
    public void setMaxIzquierda(float maxIzquierda) {
        this.maxIzquierda = maxIzquierda;
    }

    /**
     * Establece el valor de la caché de vídeo del cliente
     * @param cltCaching Valor de la caché de vídeo del cliente
     */
    public void setCltCaching(int cltCaching) {
        this.cltCaching = cltCaching;
    }

  /**
     * Establece el puerto COM de control del vehículo
     * @param COMCarrito Número de puerto COM
     */
    public void setCOMCarrito(String COMCarrito) {
        this.COMCarrito = COMCarrito;
    }

    /**
     * Establece el puerto COM de control de las cámaras
     * @param COMCamara Número de puerto COM
     */
    public void setCOMCamara(String COMCamara) {
        this.COMCamara = COMCamara;
    }

    /**
     * Establece los parámetros límite del Joystick
     * @param maxDerecha Máximo valor hacia la derecha del Joystick
     * @param maxIzquierda Máximo valor hacia la izquierda del Joystick
     * @param maxArriba Máximo valor de aceleración del Joystick
     * @param maxAbajo Máximo valor de frenado del Joystick
     * @param XNone Valor central del Joystick en el eje X
     * @param XDif Límite central dentro del cual no se puede asegurar que el
     * Joystick esté en el centro del eje X
     * @param YNone Valor central del Joystick en el eje Y
     * @param YDif Límite central dentro del cual no se puede asegurar que el
     * Joystick esté en el centro del eje Y
     */
    public void setJoystick(float maxDerecha, float maxIzquierda, float maxArriba, float maxAbajo,
                            float XNone, float XDif, float YNone, float YDif) {
        this.maxIzquierda = maxIzquierda;
        this.maxDerecha = maxDerecha;
        this.maxArriba = maxArriba;
        this.maxAbajo = maxAbajo;
        this.XNone = XNone;
        this.XDif = XDif;
        this.YNone = YNone;
        this.YDif = YDif;
    }

    /**
     * Obtiene el valor de la IP del servidor
     * @return Valor de la IP del servidor
     */
    public String getIp() {
        return ip;
    }

    /**
     * Obtiene el límite de frenado del Joystick
     * @return Valor del límite de frenado del Joystick
     */
    public float getMaxAbajo() {
        return maxAbajo;
    }

    /**
     * Obtiene el límite de aceleración del Joystick
     * @return Valor del límite de aceleración del Joystick
     */
    public float getMaxArriba() {
        return maxArriba;
    }

    /**
     * Obtiene el límite de giro hacia la derecha del Joystick
     * @return Valor del límite de giro hacia la derecha del Joystick
     */
    public float getMaxDerecha() {
        return maxDerecha;
    }

    /**
     * Obtiene el límite de giro hacia la izquierda del Joystick
     * @return Valor del límite de giro hacia la izquierda del Joystick
     */
    public float getMaxIzquierda() {
        return maxIzquierda;
    }

    /**
     * Obtiene el array con los descriptores de los emisores de vídeo
     * @return Array con los descriptores de los emisores de vídeo
     */
    public VideoSvrConf[] getEmisores() {
        return emisores;
    }

    /**
     * Obtiene el valor de caché de vídeo del cliente
     * @return Valor de caché de vídeo del cliente
     */
    public int getCltCaching() {
        return cltCaching;
    }

  /**
     * Obtiene el puerto COM de control del Vehículo
     * @return El puerto COM de control del Vehículo
     */
    public String getCOMCarrito() {
        return COMCarrito;
    }

    /**
     * Obtiene el puerto COM de control de la cámara
     * @return El puerto COM de control de la cámara
     */
    public String getCOMCamara() {
        return COMCamara;
    }

    /**
     * Obtiene el punto central del Joystick en el eje Y
     * @return Punto central del Joystick en el eje Y
     */
    public float getYNone() {
        return YNone;
    }

    /**
     * Obtiene el límite central dentro del cual no se puede asegurar que el
     * Joystick esté en el centro del eje Y
     * @return Límite central dentro del cual no se puede asegurar que el
     * Joystick esté en el centro del eje Y
     */
    public float getYDif() {
        return YDif;
    }

    /**
     * Obtiene el punto central del Joystick en el eje X
     * @return Punto central del Joystick en el eje X
     */
    public float getXNone() {
        return XNone;
    }

    /**
     * Obtiene el límite central dentro del cual no se puede asegurar que el
     * Joystick esté en el centro del eje X
     * @return Límite central dentro del cual no se puede asegurar que el
     * Joystick esté en el centro del eje X
     */
    public float getXDif() {
        return XDif;
    }

    /**
     * Comprueba si el volante ha sido calibrado o no
     * @return Devuelve true si el volante ha sido calibrado
     */
    public boolean isCalibrado() {
        if (maxIzquierda == NULLFLOAT || maxDerecha == NULLFLOAT ||
            maxArriba == NULLFLOAT || maxAbajo == NULLFLOAT) {
            return false;
        }
        return true;
    }

}
