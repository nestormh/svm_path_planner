/**
 * Paquete que contiene todas las clases descriptoras de las propiedades que
 * afectan a la aplicación
 */
package carrito.configura;

import java.awt.*;
import java.io.*;

/**
 * Clase descriptora de un receptor de vídeo en la aplicación cliente
 * @author Néstor Morales Hernández
 * @version 1.0
 */
public class VideoCltConfig implements Serializable {
    /** Dirección IP del servidor de vídeo */
    private String ip = null;
    /** Puerto del servidor de vídeo */
    private int port = Constantes.NULLINT;
    /** Valor de la caché de vídeo del cliente */
    private int caching = Constantes.NULLINT;
    /** Codec empleado para transformar el vídeo */
    private String codec = null;
    /** Bitrate empleado para transformar el vídeo */
    private int bitrate = Constantes.NULLINT;
    /** Escala empleada para transformar el vídeo */
    private float scale = Constantes.NULLFLOAT;
    /** Multiplexado empleado para transformar el vídeo */
    private String mux = null;
    /** Fichero destino de la grabación */
    private String file = null;
    /** Posición inicial del visor de vídeo */
    private Dimension posicion = null;

    /**
     * Constructor que inicializa las variables de la clase
     * @param ip Dirección IP del servidor de vídeo
     * @param port Puerto del servidor de vídeo
     * @param posicion Posición inicial del visor de vídeo
     * @param caching Valor de la caché de vídeo del cliente
     * @param codec Codec empleado para transformar el vídeo
     * @param bitrate Bitrate empleado para transformar el vídeo
     * @param scale Escala empleada para transformar el vídeo
     * @param mux Multiplexado empleado para transformar el vídeo
     * @param file Fichero destino de la grabación
     */
    public VideoCltConfig(String ip, int port, Dimension posicion, int caching,
                          String codec,
                          int bitrate, float scale, String mux, String file) {
        this.ip = ip;
        this.port = port;
        this.posicion = posicion;
        this.caching = caching;
        this.codec = codec;
        this.bitrate = bitrate;
        this.scale = scale;
        this.mux = mux;
        this.file = file;
    }

    /**
     * Obtiene el valor de Bitrate
     * @return Bitrate empleado para transformar el vídeo
     */
    public int getBitrate() {
        return bitrate;
    }

    /**
     * Obtiene el valor de la caché de vídeo del cliente
     * @return Devuelve el valor de la caché de vídeo del cliente
     */
    public int getCaching() {
        return caching;
    }

    /**
     * Obtiene el codec empleado para transformar el vídeo
     * @return Codec empleado para transformar el vídeo
     */
    public String getCodec() {
        return codec;
    }

    /**
     * Obtiene el fichero destino de la grabación
     * @return Fichero destino de la grabación
     */
    public String getFile() {
        return file;
    }

    /**
     * Obtiene la IP del servidor de vídeo
     * @return IP del servidor de vídeo
     */
    public String getIp() {
        return ip;
    }

    /**
     * Obtiene el multiplexado empleado para transformar el vídeo
     * @return Multiplexado empleado para transformar el vídeo
     */
    public String getMux() {
        return mux;
    }

    /**
     * Obtiene el puerto del servidor de vídeo
     * @return Puerto del servidor de vídeo
     */
    public int getPort() {
        return port;
    }

    /**
     * Obtiene la escala empleada para transformar el vídeo
     * @return Escala empleada para transformar el vídeo
     */
    public float getScale() {
        return scale;
    }

    /**
     * Obtiene la posición inicial del visor de vídeo
     * @return Posición inicial del visor de vídeo
     */
    public Dimension getPosicion() {
        return posicion;
    }

    /**
     * Establece el bitrate
     * @param bitrate Nuevo bitrate
     */
    public void setBitrate(int bitrate) {
        this.bitrate = bitrate;
    }

    /**
     * Establece el valor de la caché del cliente de vídeo
     * @param caching Nuevo valor de caché
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
     * Establece el fichero destino de la grabación
     * @param file Fichero destino
     */
    public void setFile(String file) {
        this.file = file;
    }

    /**
     * Establece la IP del servidor
     * @param ip Nueva IP
     */
    public void setIp(String ip) {
        this.ip = ip;
    }

    /**
     * Establece el valor de multiplexado
     * @param mux Nuevo multiplexado
     */
    public void setMux(String mux) {
        this.mux = mux;
    }

    /**
     * Establece el puerto del servidor
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
     * Establece la posición del visor de vídeo
     * @param posicion Nueva posición
     */
    public void setPosicion(Dimension posicion) {
        this.posicion = posicion;
    }

}
