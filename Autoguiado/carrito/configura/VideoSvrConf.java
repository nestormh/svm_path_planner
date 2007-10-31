/**
 * Paquete que contiene todas las clases descriptoras de las propiedades que
 * afectan a la aplicación
 */
package carrito.configura;

import java.io.*;
import java.net.*;

/**
 * Clase descriptora de un emisor de vídeo en la aplicación servidor
 * @author Néstor Morales Hernández
 * @version 1.0
 */
public class VideoSvrConf implements Serializable {
    /** Nombre del dispositivo de captura de vídeo */
    private String name = "";
    /** Número de dispositivo de captura de vídeo */
    private int videoDisp = Constantes.NULLINT;
    /** Ancho */
    private int width = 176;
    /** Alto */
    private int height = 144;
    /** Caché de captura */
    private int caching = 300;
    /** FPS */
    private float fps = 25.0f;
    /** Códec */
    private String codec = "DIV3";
    /** Bitrate */
    private int bitrate = 1024;
    /** Escala */
    private float scale = 1;
    /** Indica si se va a mostrar el vídeo en la máquina servidor o no */
    private boolean display = false;
    /** Multiplexado */
    private String mux = "ts";
    /** Fichero destino en caso que se desee grabar lo que se está transmitiendo */
    private String file = null;
    /** Dirección IP del servidor */
    private String ip = "127.0.0.1";
    /** Puerto del servidor */
    private int port = 1234;
    /** Puerto serie para modificar el Zoom de la cámara */
    private String serial = "COM3";

    /**
     * Constructor del descriptor. Establece el nombre del dispositivo asociado
     * y el número del dispositivo. Además, establece una IP y un número de puerto
     * por defecto
     * @param name Nombre del dispositivo de captura de vídeo asociado
     * @param videoDisp Número del dispositivo de captura de vídeo asociado
     */
    public VideoSvrConf(String name, int videoDisp) {
        this.name = name;
        this.videoDisp = videoDisp;
        this.port += videoDisp;
        try {
            this.ip = InetAddress.getLocalHost().getHostAddress();
        } catch(UnknownHostException uhe) {}
    }

    /**
     * Establece todos los parámetros del descriptor
     * @param width Ancho
     * @param height Alto
     * @param caching Valor de la caché respecto al dispositivo de captura
     * @param fps Frames por segundo
     * @param codec Códec
     * @param bitrate Bitrate
     * @param scale Escala
     * @param display Indica si se va a mostrar el vídeo capturado en la máquina emisora
     * @param mux Multiplexado
     * @param file Fichero, en caso que se desee registrar lo que se está grabando
     * @param ip Dirección IP del servidor
     * @param port Puerto del servidor
     * @param serial Puerto COM de control del Zoom
     */
    public void setValues(int width, int height, int caching, float fps,
                          String codec, int bitrate, float scale, boolean display, String mux,
                          String file, String ip, int port, String serial) {

        this.videoDisp = videoDisp;
        this.width = width;
        this.height = height;
        this.caching = caching;
        this.fps = fps;
        this.codec = codec;
        this.bitrate = bitrate;
        this.scale = scale;
        this.display = display;
        this.mux = mux;
        this.file = file;
        this.ip = ip;
        this.port = port;
        this.serial = serial;
    }

    /**
     * Obtiene el Bitrate
     * @return Bitrate
     */
    public int getBitrate() {
        return bitrate;
    }

    /**
     * Obtiene la caché respecto al dispositivo de captura
     * @return Tamaño de la caché
     */
    public int getCaching() {
        return caching;
    }

    /**
     * Obtiene el códec que se está empleando para transformar el vídeo al formato
     * en el cual será enviado
     * @return Códec usado
     */
    public String getCodec() {
        return codec;
    }

    /**
     * Indica si se va a mostrar el vídeo capturado en el servidor
     * @return true, en caso afirmativo
     */
    public boolean isDisplay() {
        return display;
    }

    /**
     * Obtiene el nombre del fichero en el que se va a grabar
     * @return Nombre del fichero
     */
    public String getFile() {
        return file;
    }

    /**
     * Obtiene el valor de FPS
     * @return Valor de FPS
     */
    public float getFps() {
        return fps;
    }

    /**
     * Obtiene la altura del vídeo
     * @return Altura del vídeo
     */
    public int getHeight() {
        return height;
    }

    /**
     * Obtiene la IP del servidor
     * @return IP del servidor
     */
    public String getIp() {
        return ip;
    }

    /**
     * Obtiene el valor de multiplexado
     * @return Multiplexado
     */
    public String getMux() {
        return mux;
    }

    /**
     * Obtiene el puerto del servidor
     * @return Puerto del servidor
     */
    public int getPort() {
        return port;
    }

    /**
     * Obtiene la escala del vídeo
     * @return Escala del vídeo
     */
    public float getScale() {
        return scale;
    }

    /**
     * Obtiene el número de dispositivo en el sistema
     * @return int
     */
    public int getVideoDisp() {
        return videoDisp;
    }

    /**
     * Obtiene el ancho del vídeo
     * @return Ancho del vídeo
     */
    public int getWidth() {
        return width;
    }

    /**
     * Obtiene el nombre del dispositivo
     * @return Nombre del dispositivo
     */
    public String getName() {
        return name;
    }

    /**
     * Obtiene el puerto COM que controla el zoom de la cámara
     * @return String
     */
    public String getSerial() {
        return serial;
    }

    /**
     * Establece el Bitrate
     * @param bitrate Nuevo bitrate
     */
    public void setBitrate(int bitrate) {
        this.bitrate = bitrate;
    }

    /**
     * Establece el tamaño de la caché respecto al dispositivo de captura
     * @param caching Nuevo tamaño de la caché
     */
    public void setCaching(int caching) {
        this.caching = caching;
    }

    /**
     * Establece el nuevo códec
     * @param codec Nuevo códec
     */
    public void setCodec(String codec) {
        this.codec = codec;
    }

    /**
     * Establece si el vídeo será mostrado en el servidor
     * @param display Indica si será mostrado o no
     */
    public void setDisplay(boolean display) {
        this.display = display;
    }

    /**
     * Establece el nombre del fichero en el que se grabará
     * @param file Nuevo nombre de fichero
     */
    public void setFile(String file) {
        this.file = file;
    }

    /**
     * Establece el valor de FPS
     * @param fps Nuevo valor de FPS
     */
    public void setFps(float fps) {
        this.fps = fps;
    }

    /**
     * Establece la altura del vídeo
     * @param height Nueva altura
     */
    public void setHeight(int height) {
        this.height = height;
    }

    /**
     * Establece la dirección IP del servidor
     * @param ip Nueva IP
     */
    public void setIp(String ip) {
        this.ip = ip;
    }

    /**
     * Establece el nuevo valor de multiplexado
     * @param mux Nuevo valor de multiplexado
     */
    public void setMux(String mux) {
        this.mux = mux;
    }

    /**
     * Establece el nuevo puerto del servidor
     * @param port Nuevo puerto
     */
    public void setPort(int port) {
        this.port = port;
    }

    /**
     * Establece la escala del vídeo
     * @param scale Nueva escala
     */
    public void setScale(float scale) {
        this.scale = scale;
    }

    /**
     * Establece el número del dispositivo
     * @param videoDisp Nuevo número de dispositivo
     */
    public void setVideoDisp(int videoDisp) {
        this.videoDisp = videoDisp;
    }

    /**
     * Establece el ancho del vídeo
     * @param width Nuevo ancho
     */
    public void setWidth(int width) {
        this.width = width;
    }

    /**
     * Establece el nombre del dispositivo
     * @param name Nuevo nombre
     */
    public void setName(String name) {
        this.name = name;
    }

    /**
     * Establece el puerto COM de control del Zoom de las cámaras
     * @param serial Nuevo puerto COM
     */
    public void setSerial(String serial) {
        this.serial = serial;
    }

    /**
     * Devuelve el nombre del dispositivo
     * @return Nombre del dispositivo
     */
    public String toString() {
        return name;
    }
}
