/**
 * Paquete que contiene todas las clases relacionadas con el apartado multimedia de la aplicación
 */
package carrito.media;

import carrito.configura.*;

/**
 * Clase que extiende la clase Media para adaptar la creación de instancias para
 * el cliente de la aplicación
 * @author Néstor Morales Hernández
 * @version 1.0
 */
public class ClienteMedia extends Media {
    /**
     * Constructor de la clase. Únicamente recibe la ubicación de la librería
     * VLC y se la pasa ala clase padre.
     * @param dllpath Ubicación de la librería VLC
     */
    public ClienteMedia(String dllpath) {
        super(dllpath);
    }

    /**
     * Añade una instancia para un cliente a partir de un objeto del tipo VideoCltConfig
     * @param vcc Objeto que contiene toda la descripción del cliente
     * @return Devuelve el identificador de la nueva instancia creada
     * @throws Lanza una excepción si alguno de los datos del cliente son inadecuados
     */
    public int addCliente(VideoCltConfig vcc) throws Exception {
        // Desglosa el objeto VideoCltConfig
        String ip = vcc.getIp();
        int port = vcc.getPort();
        int caching = vcc.getCaching();
        String codec = vcc.getCodec();
        int bitrate = vcc.getBitrate();
        float scale = vcc.getScale();
        String mux = vcc.getMux();
        String file = vcc.getFile();
        // Llama a la función propia addCliente con el objeto ya desglosado
        return addCliente(ip, port, caching, codec, bitrate, scale, mux, file, true);
    }

    /**
     * Genera una cadena correspondiente a un comando VLC a partir de los parámetros
     * recibidos. A partir de esta cadena, crea una nueva instancia VLC
     * @param ip IP del servidor
     * @param port Puerto del servidor
     * @param caching Tamaño del buffer de recepción
     * @param codec Códec del video recibido
     * @param bitrate Bitrate del video
     * @param scale Escala del video
     * @param mux Multiplexado
     * @param file Fichero en el que se va a guardar el video recibido
     * @param visible Indica si se va a mostrar el vídeo recibido
     * @return Identificador de la instancia creada
     * @throws Lanza una excepción si alguno de los datos es incorrecto
     */
    public int addCliente(String ip, int port, int caching, String codec,
                          int bitrate,
                          float scale, String mux, String file, boolean visible) throws
            Exception {
        String comando = "";
        if ((ip == null) || (port <= 1024)) {
            throw new Exception(
                    "IP o puerto de emisión inválidos. No se pudo continuar");
        }
        comando += "http://" + ip + ":" + port + " ";
        if (caching > Constantes.NULLINT) {
            comando += ":http-caching=" + caching + " ";
        }

        comando += ":sout=#";
        if (file != null) {
            if (codec != null) {
                int vb = 1024;
                float size = 1.0f;
                if (bitrate > Constantes.NULLINT) {
                    vb = bitrate;
                }
                if ((scale != Constantes.NULLFLOAT) && (scale >= 0.0f)) {
                    size = scale;
                }
                comando += "transcode{vcodec=" + codec + ",vb=" + vb +
                        ",scale=" + size + "}:";
            }
        }
        comando += "duplicate{";
        if (visible) {
            comando += "dst=display";
        }
        if (file != null) {
            if (mux == null) {
                throw new Exception(
                        "No se ha especificado formato de multiplexado. No se pudo continuar");
            }
            if (visible) {
                comando += ",";
            }
            comando += "dst=std{access=file,mux=" + mux + ",dst=\"" + file +
                    "\"}";
        }
        comando += "}";

        return addInstancia(comando);
    }

}
