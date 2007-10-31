/**
 * Paquete que contiene todas las clases correspondientes a la aplicación servidor
 */
package carrito.server;

import java.awt.*;
import java.rmi.*;

import carrito.configura.*;

/**
 * Interfaz que describe todos los métodos que han de ser escritos para la creación
 * del objeto RMI remoto
 * @author Néstor Morales Hernández
 * @version 1.0
 */
public interface InterfazRMI extends Remote {
    /**
     * Devuelve una lista de objetos VideoCltConfig, los cuales describen un cliente
     * multimedia que va a corresponder a cada uno de los servidores multimedia en la
     * aplicación servidor
     * @return Devuelve una lista de objetos VideoCltConfig
     * @throws RemoteException
     */
    public VideoCltConfig[] getVideos() throws RemoteException;

    /**
     * Obtiene el tamaño del vídeo transmitido desde el dispositivo indicado
     * @param cam Número de dispositivo al que se hace referencia
     * @return Devuelve el tamaño de vídeo solicitado
     * @throws RemoteException
     */
    public Dimension getSize(int cam) throws RemoteException;

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
    public boolean avanzaCarro(int own, float aceleracion, float frenado,
                               float giro) throws RemoteException;

    /**
     * Indica el ángulo lateral de las cámaras
     * @param camara Cámara a la que se hace referencia
     * @param angulo Ángulo indicado
     * @throws RemoteException
     */
    public void setAnguloCamaras(int camara, int angulo) throws RemoteException;

    /**
     * Indica el ángulo en altura de las cámaras
     * @param angulo Ángulo indicado
     * @throws RemoteException
     */
    public void setAlturaCamaras(int angulo) throws RemoteException;

    /**
     * Indica el zoom de una determinada cámara
     * @param zoom Indica el nuevo zoom
     * @param id Cámara a la que se hace referencia
     * @throws RemoteException
     */
    public void setZoom(int zoom, int id) throws RemoteException;

    /**
     * Solicita el control del vehículo
     * @return Devuelve -1 si se deniega el control, u otro número en caso de que
     * este haya sido concedido, indicando el identificador de posesión del control
     * @throws RemoteException
     */
    public int getJoystick() throws RemoteException;

    /**
     * Libera el control del vehículo
     * @param id Identificador de posesión del vehículo
     * @throws RemoteException
     */
    public void freeJoystick(int id) throws RemoteException;
    
    public boolean frenoTotal(int id) throws RemoteException;
    public boolean desfrenoTotal(int id) throws RemoteException;
    public boolean resetAvance(int id, boolean retrocede) throws RemoteException;
    
    
}
