/**
 * Paquete que contiene todas las clases que son específicas del cliente
 */
package carrito.cliente;

import java.rmi.*;

import javax.swing.*;

import carrito.cliente.interfaz.*;
import carrito.cliente.interfaz.opciones.*;
import carrito.configura.*;
import carrito.media.*;
import carrito.server.*;

/**
 * Clase principal del cliente. Inicia todas las partes que componen la
 * aplicación y establece la comunicación con el servidor
 * @author Néstor Morales Hernández
 * @version 1.0
 */
public class Cliente extends JFrame {
    /** Objeto que hace de interfaz entre todas las variables comunes a la aplicación */
    private Constantes cte = null;
    /** Objeto multimedia */
    private ClienteMedia media = null;
    /** Objeto RMI */
    private InterfazRMI rmi = null;
    /** Ventana principal de la aplicación */
    private frmCliente form = null;
    /** Diálogo de control de las cámaras */
    private DlgCamaras camaras = null;

    /**
     * Constructor de la clase cliente. Inicializa el objeto Constantes, obtiene
     * la configuración del fichero de configuración, muestra la ventana de
     * configuración, inicia las distintas partes de la aplicación y muestra la
     * interfaz de usuario.
     * @param dllpath Ubicación de la librería VLC
     */
    public Cliente(String dllpath) {
        // Inicializa el objeto constantes
        cte = new Constantes();
        // Abre la configuración desde un fichero
        cte.openCliente();

        // Establece el icono que van a heredar las ventanas asociadas
        setIconImage(new ImageIcon("cars.jpg").getImage());
        // Abre la ventana de configuración
        DlgConfiguraClt dcc = new DlgConfiguraClt(this, cte);
        setResizable(false);
        dcc.setVisible(true);
        while (dcc.isVisible()) {
            try {
                Thread.sleep(100);
            } catch (Exception e) {}
        }

        // Si es la primera vez que se abre la aplicación o el volante no está
        // calibrado, inicia el asistente de calibrado
        if (! cte.isCalibrado()) {
            DlgOptJoystick doj = new DlgOptJoystick(this, cte);
            doj.show();
        }

        // Inicia la comunicación con el servidor RMI
        iniciaRMI();
        // Inicia el objeto multimedia y las instancias que van a leer el stream remoto
        iniciaMedia(dllpath);
        // Inicia la interfaz
        iniciaInterfaz();
    }

    /**
     * Inicia la interfaz de la aplicación
     */
    private void iniciaInterfaz() {
        // Crea la ventana de control de las cámaras
        camaras = new DlgCamaras(cte, media, rmi);
        // Crea la ventana principal
        form = new frmCliente(cte, rmi, camaras);
        form.show();
    }

    /**
     * Crea el objeto multimedia, asi como una instancia para cada una de las
     * cámaras que están enviando video desde el servidor
     * @param dllpath Ubicación de la librería VLC. Es necesario indicarlo, ya que ha
     * de ser el primer parámetro en la cadena que maneja esta librería para
     * crear las instancias multimedia
     */
    private void iniciaMedia(String dllpath) {
        // Crea el objeto multimedia
        media = new ClienteMedia(dllpath);
        // Crea los objetos descriptores de la recepción multimedia a partir de
        cte.setReceptores(rmi);
        // Crea una instancia para cada descriptor y almacena su identificador en un
        // array que va a ser incluido en el objeto constantes
        try {
            int idVideos[] = new int[cte.getNumReceptores()];
            for (int i = 0; i < cte.getNumReceptores(); i++) {
                idVideos[i] = media.addCliente(cte.getReceptor(i));
            }
            cte.setIdVideos(idVideos);
        } catch(Exception e) {
            Constantes.mensaje("No se pudo crear instancia de vídeo", e);
        }
    }

    /**
     * Inicia la comunicación RMI con el objeto <i>ServidorCarrito</i>
     */
    private void iniciaRMI() {
        String url = "rmi://" + cte.getIp() + "/ServidorCarrito";
        try {
            rmi = (InterfazRMI) Naming.lookup(url);
        } catch (Exception e) {
            Constantes.mensaje("No se pudo obtener el objeto RMI. ", e);
        }
    }

    /**
     * Método main del cliente. Arranca el cliente y obtiene el primer argumento,
     * que ha de indicar la ubicación de la librería VLC. Si no existe este argumento, devuelve
     * un error indicando el uso adecuado de la aplicación
     * @param args Argumentos
     */
    public static void main(String[] args) {
        if (args.length != 1) {
            System.err.println("Faltan argumentos:\n\tUso: $>java [opciones] carrito.cliente.Cliente <ubicación de la librería VLC>");
            System.exit(1);
        }
        Cliente cliente = new Cliente(args[0]);
    }
}
