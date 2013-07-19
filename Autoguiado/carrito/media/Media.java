/**
 * Paquete que contiene todas las clases relacionadas con el apartado multimedia de la aplicación
 */
package carrito.media;

import java.util.*;
import java.awt.Canvas;

/**
 * Clase que automatiza todas las tareas multimedia mediante la creación de una instancia
 * a partir de una serie de comandos VLC y el control posterior de dichas instancias. Muchas de
 * las funciones no se emplean en la aplicación actual, pero se ha dejado preparado para su uso
 * en cualquier aplicación multimedia, permitiendo así usar todo el potencial de VLC en aplicaciones
 * Java.
 * @author Néstor Morales Hernández
 * @version 1.0
 */
public class Media {

    // La librería es cargada al crearse la clase
    static {
        System.loadLibrary("javavlc");
    }

    private native int _endMC();
    private native int _getAncho(String nombre);
    private native int _getAlto(String nombre);
    private native int[] _getImagen(String nombre);
    public native void _saveImagenActual(String dispositivo, String nombre);
    public native void _loadImagen(String nombre);

    /**
     * Método nativo que carga la librería VLC (NOTA: En este caso nos referimos
     * a la librería usada por el programa VLC original que hemos usado para
     * crear la librería propia
     * @return Devuelve <i>true</i> si la librería se cargó sin problemas
     */
    private native boolean _cargaLibreria(String dllpath);

    /**
     * Método nativo que crea una instancia multimedia a partir de una serie de
     * parámetros incluidos en un array de Sring
     * @param cadena Array con los parámetros especificados
     * @return Devuelve el número identificador de la instancia creada
     */
    private native int _creaInstancia(String cadena[]);

    /**
     * Método nativo que inicia la reproducción de una determinada instancia multimedia
     * @param id Número identificador de la instancia multimedia que se quiere
     * reproducir
     * @return Devuelve <i>true</i> si no ha habido problemas
     */
    private native boolean _play(int id);

    /**
     * Método nativo que pausa una determinada instancia multimedia
     * @param id Número identificador de la instancia multimedia que se quiere
     * pausar
     * @return Devuelve <i>true</i> si no ha habido problemas
     */
    private native boolean _pausa(int id);

    /**
     * Método nativo que detiene una determinada instancia multimedia
     * @param id Número identificador de la instancia multimedia que se quiere
     * detener
     * @return Devuelve <i>true</i> si no ha habido problemas
     */
    private native boolean _stop(int id);

    /**
     * Método nativo que reproduce una instancia determinada a pantalla completa
     * @param id Número identificador de la instancia multimedia que se quiere
     * reproducir
     * @return v
     */
    private native boolean _fullScreen(int id);

    /**
     * Método nativo que comprueba si se está reproduciendo una instancia multimedia
     * @param id Número identificador de la instancia multimedia que se quiere
     * comprobar
     * @return Devuelve <i>true</i> si se está reproduciendo
     */
    //private native boolean _isPlaying(int id);

    /**
     * Método nativo que obtiene la longitud del stream correspondiente a una determinada instancia
     * multimedia
     * @param id Número identificador de la instancia multimedia que se quiere
     * comprobar
     * @return Devuelve la longitud del stream
     */
    private native int _getLength(int id);

    /**
     * Método nativo que vacía el playlist de una determinada instancia multimedia
     * @param id Número identificador de la instancia multimedia que se quiere
     * vaciar
     * @return Devuelve <i>true</i> si no ha habido problemas
     */
    private native boolean _clearPlaylist(int id);

    /**
     * Método nativo que obtiene el índice en la lista de reproducción actual para una determinada
     * instancia multimedia
     * @param id Número identificador de la instancia multimedia que se quiere
     * comprobar
     * @return Índice actual en la lista de reproducción
     */
    private native int _getPlaylistIndex(int id);

    /**
     * Método nativo que reproduce el siguiente stream en una determinada lista de reproducción para
     * una determinada instancia
     * @param id Número identificador de la instancia multimedia sobre la que se quiere
     * actuar
     * @return Devuelve <i>true</i> si no ha habido problemas
     */
    private native boolean _nextPlaylist(int id);

    /**
     * Método nativo que reproduce el stream anterior en una determinada lista de reproducción para
     * una determinada instancia
     * @param id Número identificador de la instancia multimedia sobre la que se quiere
     * actuar
     * @return Devuelve <i>true</i> si no ha habido problemas
     */
    private native boolean _lastPlaylist(int id);

    /**
     * Método nativo que obtiene la longitud de la lista de reproducción para una instancia determinada
     * @param id Número identificador de la instancia multimedia sobre la que se quiere
     * actuar
     * @return Longitud de la lista de reproducción
     */
    private native int _getPlayListLength(int id);

    /**
     * Método nativo que obtiene la posición actual (entre 0 y 1) dentro del stream
     * @param id Número identificador de la instancia multimedia sobre la que se quiere
     * actuar
     * @return Posición actual
     */
    private native float _getPos(int id);

    /**
     * Método nativo que establece la posición (entre 0 y 1) dentro del stream
     * @param id Número identificador de la instancia multimedia sobre la que se quiere
     * actuar
     * @param pos Posición que se desea establecer
     * @return Devuelve <i>true</i> si no ha habido problemas
     */
    private native boolean _setPos(int id, float pos);

    /**
     * Método nativo que acelera la velocidad de reproducción
     * @param id Número identificador de la instancia multimedia sobre la que se quiere
     * actuar
     * @return Devuelve <i>true</i> si no ha habido problemas
     */
    private native boolean _setFaster(int id);

    /**
     * Método nativo que ralentiza la velocidad de reproducción
     * @param id Número identificador de la instancia multimedia sobre la que se quiere
     * actuar
     * @return Devuelve <i>true</i> si no ha habido problemas
     */
    private native boolean _setSlower(int id);

    /**
     * Método nativo que obtiene el tiempo actual de reproducción en segundos
     * @param id Número identificador de la instancia multimedia sobre la que se quiere
     * actuar
     * @return Tiempo de reproducción actual desde el inicio de la reproducción (en segundos)
     */
    private native int _getTime(int id);

    /**
     * Método nativo que establece el tiempo actual de reproducción en segundos
     * @param id Número identificador de la instancia multimedia sobre la que se quiere
     * actuar
     * @param seconds Posición de tiempo en la que se desea pasar
     * @param relative Si está a <i>true</i>, se empezará a contar a partir de la posición actual
     * @return Devuelve <i>true</i> si no ha habido problemas
     */
    private native boolean _setTime(int id, int seconds, boolean relative);

    /**
     * Método nativo que obtiene el volumen del sonido actual
     * @param id Número identificador de la instancia multimedia sobre la que se quiere
     * actuar
     * @return Valor del volumen actual
     */
    private native int _getVolume(int id);

    /**
     * Método nativo que establece el volumen del sonido actual
     * @param id Número identificador de la instancia multimedia sobre la que se quiere
     * actuar
     * @param volume Valor del volumen que se desea establecer
     * @return Devuelve <i>true</i> si no ha habido problemas
     */
    private native boolean _setVolume(int id, int volume);

    /**
     * Método nativo que silencia el stream. Equivale a setVolume(0)
     * @param id Número identificador de la instancia multimedia sobre la que se quiere
     * actuar
     * @return Devuelve <i>true</i> si no ha habido problemas
     */
    private native boolean _setMute(int id);

    /**
     * Método nativo que elimina una instancia de VLC
     * @param id Número identificador de la instancia multimedia que se desea eliminar
     * @return Devuelve <i>true</i> si no ha habido problemas
     */
    private native boolean _eliminaInstancia(int id);

    /**
     * Método nativo que libera la librería VLC (NOTA: En este caso nos referimos
     * a la librería usada por el programa VLC original que hemos usado para
     * crear la librería propia
     * @return Devuelve <i>true</i> si la librería se liberó sin problemas
     */
    private native boolean _liberaLibreria();

    /**
     * Método nativo que devuelve un array con todos los dispositivos conectados actualmente
     * @return Devuelve un array con el nombre de los todos los dispositivos
     * conectados actualmente de la forma: "<dispositivo>:<Número de dispositivo>"
     */
    private native static String[] _listaDispositivos();

    /** Lista de los dispositivos conectados actualmente */
    private String dispositivos[] = null;
    /** Vector de instancias VLC creadas (contiene los identificadores) */
    private Vector instancias = null;
    /** Path para acceder a la librería VLC */
    private String dllpath = "";

    /**
     * Constructor de la clase. Obtiene la ubicación de la librería VLC y carga la
     * librería propia. Inicializa el vector de dispositivos conectados y cambia los
     * espacios por &nbsp;, para poder separar los comandos en cadenas separadas posteriormente.
     * También crea el vector de instancias
     * @param dllpath Path de la librería VLC
     */
    public Media(String dllpath) {
        this.dllpath = dllpath;

        // Carga la librería JNI que hemos creado
        System.out.println("Libreria " + dllpath);
        if (!_cargaLibreria(dllpath)) {
            System.err.println("Error al cargar la libreria");
            System.exit( -1);
        }
        // Obtiene la lista de dispositivos conectados
        dispositivos = listaDispositivos();

        // Transforma los espacios en &nbsp;
        if (dispositivos != null) {
            for (int i = 0; i < dispositivos.length; i++) {
                dispositivos[i] = dispositivos[i].replaceAll(" ", "&nbsp;");
            }
        }
        // Crea un vector vacío de instancias
        instancias = new Vector();
    }

    /**
     * Destructor de la clase. Elimina todas las instancias y libera la librería
     * @return Devuelve <i>true</i> si todo ha ido bien
     */
    public boolean Destruye() {
        boolean isOk = true;

        // Elimina todas las instancias
        for (int i = 0; i < instancias.size(); i++) {
            if (!_eliminaInstancia(((Integer) instancias.elementAt(i)).intValue())) {
                isOk = false;
            }
        }

        // Libera la librería VLC
        if (!_liberaLibreria()) {
            isOk = false;
        }

        this._endMC();

        return isOk;
    }

    /**
     * Hace una llamada al método nativo _listaDispositivos() y obtiene la lista
     * de dispositivos conectados
     * @return Lista de dispositivos conectados
     */
    public static String[] listaDispositivos() {
        return _listaDispositivos();
    }

    /**
     * Utilidad para transformar una cadena en un array de Strings, que será pasada
     * como argumentos a la librería JNI
     * @param cadena Cadena a transformas
     * @return Array de String resultante
     */
    private String[] parser(String cadena) {
        String patrones[] = cadena.split(" ");
        String retorno[] = new String[patrones.length + 1];
        // El primer parámetro siempre ha de ser la ubicación de la librería, si no
        // dará error
        retorno[0] = dllpath;
        for (int i = 0; i < patrones.length; i++) {
            retorno[i + 1] = patrones[i].replaceAll("&nbsp;", " ");
        }
        return retorno;
    }

    /**
     * Añade una instancia a partir de un comando
     * @param comando Comando indicado
     * @return Identificador de la instancia creada
     */
    public int addInstancia(String comando) {
      System.out.println(parser(comando)[2]);
        int id = -1;
        // Crea la instancia con el comando dividido en subcadenas
        id = _creaInstancia(parser(comando));

        // Añade el identificador a la lista de instancias
        instancias.add(new Integer(id));

        // Devuelve el identificador de la instancia
        return id;
    }

    /**
     * Añade una instancia a partir de un array de String
     * @param args Array de String con los distintos comandos
     * @return Devuelve el identificador de la instancia
     */
    public int addInstancia(String args[]) {
        String args2[] = new String[args.length + 1];
        // El primer parámetro siempre ha de ser la ubicación de la librería, si no
        // dará error
        args2[0] = dllpath;
        for (int i = 0; i < args.length; i++) {
            args2[i + 1] = args[i];
        }
        // Crea la instancia
        int id = _creaInstancia(args2);
        // Añade el identificador a la lista de instancias
        instancias.add(new Integer(id));
        // Devuelve el identificador de la instancia
        return id;
    }

    /**
     * Reproduce todas las instancias creadas a la vez
     * @return Devuelve <i>true</i> si todo ha ido bien
     */
    public boolean playAll() {
        boolean retorno = true;

        // Recorre todas las instancias e inicia su reproducción
        for (int i = 0; i < instancias.size(); i++) {
            if (!play(((Integer) instancias.elementAt(i)).intValue())) {
                retorno = false;
            }
        }
        return retorno;
    }

    /**
     * Detiene todas las instancias creadas a la vez
     * @return Devuelve <i>true</i> si todo ha ido bien
     */
    public boolean stopAll() {
        boolean retorno = true;

        for (int i = 0; i < instancias.size(); i++) {
            if (!stop(((Integer) instancias.elementAt(i)).intValue())) {
                retorno = false;
            }
        }
        return retorno;
    }

    /**
     * Pausa todas las instancias creadas a la vez
     * @return Devuelve <i>true</i> si todo ha ido bien
     */
    public boolean pauseAll() {
        boolean retorno = true;

        for (int i = 0; i < instancias.size(); i++) {
            if (!stop(((Integer) instancias.elementAt(i)).intValue())) {
                retorno = false;
            }
        }
        return retorno;
    }

    /**
     * Inicia la reproducción
     * @param id Número de instancia
     * @return Devuelve <i>true</i> si todo ha ido bien
     */
    public boolean play(int id) {
        return _play(id);
    }

    /**
     * Detiene la reproducción
     * @param id Número de instancia
     * @return Devuelve <i>true</i> si todo ha ido bien
     */
    public boolean stop(int id) {
        return _stop(id);
    }

    /**
     * Hace que la reproducción se realice a pantalla completa
     * @param id Número de instancia
     * @return Devuelve <i>true</i> si todo ha ido bien
     */
    public boolean fullScreen(int id) {
        return _fullScreen(id);
    }
    /**
     * Pausa la reproducción
     * @param id Número de instancia
     * @return Devuelve <i>true</i> si todo ha ido bien
     */
    public boolean pausa(int id) {
        return _pausa(id);
    }

    /**
     * Comprueba si se está reproduciendo
     * @param id Número de instancia
     * @return Devuelve <i>true</i> si se está reproduciendo actualmente
     */
    /*public boolean isPlaying(int id) {
        return _isPlaying(id);
    }*/
    /**
     * Obtiene el tamaño del stream
     * @param id Número de instancia
     * @return Devuelve el tamaño del stream
     */
    public int getLength(int id) {
        return _getLength(id);
    }

    /**
     * Vacía la lista de reproducción
     * @param id Número de instancia
     * @return Devuelve <i>true</i> si todo ha ido bien
     */
    public boolean clearPlaylist(int id) {
        return _clearPlaylist(id);
    }

    /**
     * Obtiene el índice actual dentro de la lista de reproducción
     * @param id Número de instancia
     * @return Devuelve el índice actual dentro de la lista de reproducción
     */
    public int getPlaylistIndex(int id) {
        return _getPlaylistIndex(id);
    }

    /**
     * Cambia al siguiente stream dentro de la lista de reproducción
     * @param id Número de instancia
     * @return Devuelve <i>true</i> si todo ha ido bien
     */
    public boolean nextPlaylist(int id) {
        return _nextPlaylist(id);
    }

    /**
     * Cambia al stream anterior dentro de la lista de reproducción
     * @param id Número de instancia
     * @return Devuelve <i>true</i> si todo ha ido bien
     */
    public boolean lastPlaylist(int id) {
        return _lastPlaylist(id);
    }

    /**
     * Obtiene el tamaño de la lista de reproducción
     * @param id Número de instancia
     * @return Devuelve <i>true</i> si todo ha ido bien
     */
    public int getPlaylistLength(int id) {
        return _getPlayListLength(id);
    }

    /**
     * Obtiene la posición dentro del stream
     * @param id Número de instancia
     * @return Un número entre 0.0f y 1.0f indicando la posición proporcional dentro del stream
     */
    public float getPos(int id) {
        return _getPos(id);
    }

    /**
     * Fija una nueva posición entre 0.0f y 1.0f dentro del stream actual
     * @param id Número de instancia
     * @param pos Nueva posición
     * @return Devuelve <i>true</i> si todo ha ido bien
     */
    public boolean setPos(int id, float pos) {
        return _setPos(id, pos);
    }

    /**
     * Hace que la reproducción se acelere
     * @param id Número de instancia
     * @return Devuelve <i>true</i> si todo ha ido bien
     */
    public boolean setFaster(int id) {
        return _setFaster(id);
    }

    /**
     * Hace que la reproducción se ralentice
     * @param id Número de instancia
     * @return Devuelve <i>true</i> si todo ha ido bien
     */
    public boolean setSlower(int id) {
        return _setSlower(id);
    }

    /**
     * Obtiene la posición absoluta dentro del stream (en segundos)
     * @param id Número de instancia
     * @return Devuelve la posición absoluta dentro del stream (en segundos)
     */
    public int getTime(int id) {
        return _getTime(id);
    }

    /**
     * Fija la posición absoluta dentro del stream (en segundos)
     * @param id Número de instancia
     * @param seconds Nueva posición
     * @param relative Indica si la nueva posición se va a fijar a partir de la actual
     * @return Devuelve <i>true</i> si todo ha ido bien
     */
    public boolean setTime(int id, int seconds, boolean relative) {
        return _setTime(id, seconds, relative);
    }

    /**
     * Obtiene el volumen actual del audio
     * @param id Número de instancia
     * @return Devuelve el volumen actual del audio
     */
    public int getVolume(int id) {
        return _getVolume(id);
    }

    /**
     * Establece un nuevo volumen de sonido
     * @param id Número de instancia
     * @param volume Nuevo volumen
     * @return Devuelve <i>true</i> si todo ha ido bien
     */
    public boolean setVolume(int id, int volume) {
        return _setVolume(id, volume);
    }

    /**
     * Silencia el sonido
     * @param id Número de instancia
     * @return Devuelve <i>true</i> si todo ha ido bien
     */
    public boolean setMute(int id) {
        return _setMute(id);
    }

    /**
     * Getter de la propiedad <i>dispositivos</i>
     * @param i Número de dispositivo que se quiere conocer
     * @return Valor del dispositivo <i>i</i>
     */
    public String getDispositivo(int i) {
        return dispositivos[i];
    }

    public int getAncho(String dispositivo) {
      return _getAncho(dispositivo);
    }

    public int getAlto(String dispositivo) {
      return _getAlto(dispositivo);
    }

    public int[] getImagen(String dispositivo) {
      return _getImagen(dispositivo);
    }

    public void saveImagenActual(String dispositivo, String nombre) {
      _saveImagenActual(dispositivo, nombre);
    }

    public void loadImagen(String nombre) {
      _loadImagen(nombre);
    }
  }
