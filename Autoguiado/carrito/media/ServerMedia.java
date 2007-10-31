/**
 * Paquete que contiene todas las clases relacionadas con el apartado multimedia de la aplicación
 */
package carrito.media;

import carrito.configura.*;

/**
 * Clase que extiende la clase Media para adaptar la creación de instancias para
 * el servidor de la aplicación
 * @author Néstor Morales Hernández
 * @version 1.0
 */
public class ServerMedia extends Media {
    /**
     * Constructor de la clase. Únicamente recibe la ubicación de la librería
     * VLC y se la pasa ala clase padre.
     * @param dllpath Ubicación de la librería VLC
     */
    public ServerMedia(String dllpath) {
        super(dllpath);
    }

    /**
     * Añade una instancia para un servidor a partir de un objeto del tipo VideoSvrConf
     * @param vsc Objeto que contiene toda la descripción del servidor
     * @return Devuelve el identificador de la nueva instancia creada
     * @throws Lanza una excepción si alguno de los datos del servidor son inadecuados
     */
    public int addServidor(VideoSvrConf vsc)  throws Exception {
        // Desglosa el objeto VideoSvrConf
        int videoDisp = vsc.getVideoDisp();
        int width= vsc.getWidth();
        int height = vsc.getHeight();
        int caching = vsc.getCaching();
        float fps = vsc.getFps();
        String codec = vsc.getCodec();
        int bitrate = vsc.getBitrate();
        float scale = vsc.getScale();
        boolean display = vsc.isDisplay();
        String mux = vsc.getMux();
        String file = vsc.getFile();
        String ip = vsc.getIp();
        int port = vsc.getPort();
        // Llama a la función propia addServidor con el objeto ya desglosado
        return addServidor(videoDisp, width, height, caching, null, fps, codec,
                    bitrate, scale, display, mux, file, ip, port);
    }

    /**
     * Genera una cadena correspondiente a un comando VLC a partir de los parámetros
     * recibidos. A partir de esta cadena, crea una nueva instancia VLC
     * @param videoDisp Número de dispositivo
     * @param width Ancho
     * @param height Alto
     * @param caching Tamaño de buffer de captura
     * @param croma Croma
     * @param fps FPS
     * @param codec Códec
     * @param bitrate Bitrate
     * @param scale Escala
     * @param display Indica si se va a mostrar o no el vídeo recibido
     * @param mux Multiplexado
     * @param file Fichero donde se va a guardar el vídeo capturado
     * @param ip IP del servidor
     * @param port Puerto del servidor
     * @return Identificador de la instancia creada
     * @throws Lanza una excepción si alguno de los datos es incorrecto
     */
    public int addServidor(int videoDisp, int width, int height, int caching,
                           String croma, float fps,
                           String codec, int bitrate, float scale,
                           boolean display, String mux,
                           String file, String ip, int port) throws Exception {
        String comando = "";
        if (videoDisp < Constantes.NULLINT) {
            throw new Exception(
                    "No ha indicado un dispositivo. No se puede continuar");
        }

        comando += "dshow:// :dshow-vdev=" + getDispositivo(videoDisp) + ":" +
                videoDisp + " ";

        if ((width > Constantes.NULLINT + 1) && (height > Constantes.NULLINT + 1)) {
            comando += ":dshow-adev=none :dshow-size=" + width + "x" +
                    height + " ";
        }

        if (caching > Constantes.NULLINT) {
            comando += ":dshow-caching=" + caching + " ";
        }

        if (croma != null) {
            comando += ":dshow-chroma=\"" + croma + "\" ";
        }

        if ((fps != Constantes.NULLFLOAT) && (fps >= 0.0f)) {
            comando += ":dshow-fps=" + fps + " ";
        }

        comando += ":sout=#";

        if (codec != null) {
            int vb = 1024;
            float size = 1.0f;
            if (bitrate > Constantes.NULLINT) {
                vb = bitrate;
            }
            if ((scale != Constantes.NULLFLOAT) && (scale >= 0.0f)) {
                size = scale;
            }
            comando += "transcode{vcodec=" + codec + ",vb=" + vb + ",scale=" +
                    size + "}:";
        }

        comando += "duplicate{";

        if (display) {
            comando += "dst=display,";
        }

        if (mux == null) {
            throw new Exception(
                    "No se ha especificado formato de multiplexado. No se pudo continuar");
        }

        if (file != null) {
            comando += "dst=std{access=file,mux=" + mux + ",dst=\"" + file +
                    "\"},";
        }

        if ((ip == null) || (port <= 1024)) {
            throw new Exception(
                    "IP o puerto de emisión inválidos. No se pudo continuar");
        }

        comando += "dst=std{access=http,mux=" + mux + ",dst=" + ip + ":" + port +
                "}} ";

        return addInstancia(comando);

    }

}
