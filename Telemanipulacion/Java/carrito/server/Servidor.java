/**
 * Paquete que contiene todas las clases correspondientes a la aplicación servidor
 */
package carrito.server;

import java.rmi.*;

import javax.swing.*;

import carrito.configura.*;
import carrito.media.*;
import carrito.server.interfaz.*;
import carrito.server.interfaz.opciones.*;
import carrito.server.serial.*;

/**
 * Clase principal del Servidor. Inicializa todos los componentes del servidor
 * y queda a la espera de las peticiones del cliente
 * @author Néstor Morales Hernández
 * @version 1.0
 */
public class Servidor {
    /** Objeto que hace de interfaz entre todas las variables comunes a la aplicación */
    private Constantes cte = null;
    /** Objeto multimedia */
    private ServerMedia media = null;
    /** Objeto de control del vehículo */
    private ControlCarro control = null;
    /** Objeto de control de las cámaras */
    private ControlCamara camaras = null;
    /** Objeto de control del zoom */
    private ControlZoom zoom = null;

    /**
     * Constructor del servidor. Crea el objeto que contendrá todas las variables
     * comunes a la aplicación. Posteriormente crea la ventana de configuración de
     * dichas variables, a partir de las cuales iniciará la comunicación con los puertos
     * COM, la creación de los servidores multimedia y el servidor RMI
     * @param dllpath Ubicación de la librería VLC
     */
    public Servidor(String dllpath) {
        cte = new Constantes();
        DlgConfiguraSvr dcs = new DlgConfiguraSvr(cte);
        dcs.setVisible(true);
        while (dcs.isVisible()) {
            try {
                Thread.sleep(100);
            } catch (Exception e) {}
        }

        DlgConfiguraOpciones dco = new DlgConfiguraOpciones(cte);
        dco.setVisible(true);
        while (! dco.isInicializado()) {
            try {
                Thread.sleep(100);
            } catch (Exception e) {}
        }

        initSerial();
        initMedia(dllpath);
        initRMI();
    }
    /**
     * Inicializa los objetos encargados de la comunicación por los puertos COM
     */
    private void initSerial() {
        control = new ControlCarro(cte);
        //camaras = new ControlCamara(cte);
        //zoom = new ControlZoom(cte);
    }

    /**
     * Inicializa el servidor RMI ubicando un objeto bajo el nombre
     * <i>ServidorCarrito</i>
     */
    private void initRMI() {
        // Crea el objeto RMI
        try {
            ServidorRMI rmi = new ServidorRMI(cte, control, camaras, zoom);
            Naming.bind("ServidorCarrito", rmi);
        } catch (Exception e) {
            System.out.println("Excepción: " + e.getMessage());
            System.out.println("Error al crear objeto RMI");
            System.exit(-1);
        }
        // Ubica el objeto RMI en el servidor
        System.out.println("Objeto RMI creado");
        try {
            String[] bindings = Naming.list( "" );
            System.out.println( "Vínculos disponibles:");
            for ( int i = 0; i < bindings.length; i++ )
                System.out.println( bindings[i] );
        } catch (Exception e) {
            System.out.println("Excepción: " + e.getMessage());
            System.out.println("Error al obtener vínculos RMI disponibles");
            System.exit(-1);
        }
    }

    /**
     * Inicializa el servidor multimedia
     * @param dllpath Ubicación de la librería VLC
     */
    private void initMedia(String dllpath) {
        // Crea el objeto multimedia
        media = new ServerMedia(dllpath);
        // Recorre la lista de descriptores de los servidores y añade una
        // instancia multimedia por cada uno de ellos
        for (int i = 0; i < cte.getNumEmisores(); i++) {
            try {
                int id = media.addServidor(cte.getEmisor(i));
                // Si está activada la opción de mostrar, crea un visor
                if (cte.getEmisor(i).isDisplay()) {
                        // Se crea un frame invisible que hará de owner. Esto únicamente se hace para
                        // poder modificar el icono del diálogo, el cual hereda de su owner
                        JFrame icono = new JFrame();
                        icono.setVisible(false);
                        icono.setIconImage(new ImageIcon("cars.jpg").getImage());
                        new Visor(icono, media, id, cte.getEmisor(i));
                }
            } catch(Exception e) {
                System.err.println("Excepción: " + e.getMessage());
                System.err.println("No se pudo crear el emisor de video " + i);
            }
        }
        //media.addInstancia("C:\\Documents&nbsp;and&nbsp;Settings\\neztol\\Escritorio\\Viaje&nbsp;La&nbsp;Graciosa&nbsp;'06\\Graciosa.wmv " +
        //                   ":sout=#transcode{vcodec=DIV3,vb=1024,scale=1}:duplicate{dst=std{access=http,mux=ts,dst=192.168.1.37:1234}}");

        // Reproduce todos los streams
        media.playAll();
    }

    /**
     * Método main del servidor. Arranca el servidor y obtiene el primer argumento,
     * que ha de indicar la ubicación de la librería VLC. Si no existe este argumento, devuelve
     * un error indicando el uso adecuado de la aplicación
     * @param args Argumentos
    */
    public static void main(String[] args) {
        if (args.length != 1) {
            System.err.println("Faltan argumentos:\n\tUso: $>java [opciones] carrito.server.Servidor <ubicación de la librería VLC>");
            System.exit(1);
        }
        Servidor servidor = new Servidor(args[0]);
    }
}
