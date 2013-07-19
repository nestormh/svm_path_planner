/**
 * Paquete que contiene las clases correspondientes a la interfaz del cliente
 */
package carrito.cliente.interfaz;

import java.awt.*;
import java.awt.event.*;
import java.rmi.*;

import javax.swing.*;
import javax.swing.event.*;

import carrito.configura.*;
import carrito.media.*;
import carrito.server.*;

/**
 * Clase que crea la interfaz en la que se van a incluir los paneles que
 * controlan cada una de las cámaras. Este número de paneles va a ser variable
 * según el número de retransmisiones que se estén efectuando desde el servidor.
 * @author Néstor Morales Hernández
 * @version 1.0
 */
public class DlgCamaras extends JDialog implements ChangeListener, WindowListener {
    /** Ancho predefinido de la ventana (No es redimensionable) */
    private static final int ANCHO = 280;

    // Variables de la interfaz
    private JLabel lblCamaras = new JLabel("Ángulo en altura: ");
    private JSpinner spnCamaras = new JSpinner(new SpinnerNumberModel(0, -90, 90, 0.5f));
    private Menu menu = null;

    /** Objeto RMI */
    private InterfazRMI rmi = null;

    /** Lista de paneles incluidos en la clase */
    private PnlCamara paneles[] = null;

    /** Objeto que hace de interfaz entre todas las variables comunes a la aplicación */
    private Constantes cte = null;

    /**
     * Constructor de la clase. Obtiene el número de vídeos que se emiten y crea
     * un vector de Paneles de ese tamaño. Luego crea un panel para cada uno de los
     * streams recibidos
     * @param cte Objeto que hace de interfaz entre todas las variables comunes a la aplicación
     * @param media Objeto multimedia que será pasado a cada uno de los paneles
     * @param rmi Objeto RMI
     */
    public DlgCamaras(Constantes cte, ClienteMedia media, InterfazRMI rmi) {
        this.rmi = rmi;
        this.cte = cte;
        this.setTitle("Control de las Cámaras");
        this.getContentPane().setLayout(new FlowLayout());
        this.menu = menu;

        getContentPane().add(lblCamaras);
        getContentPane().add(spnCamaras);
        // Crea un vector del tamaño "cte.getNumReceptores()"
        paneles = new PnlCamara[cte.getNumReceptores()];
        // Crea el número de paneles indicado por la función anterior
        for (int i = 0; i < cte.getNumReceptores(); i++) {
            paneles[i] = new PnlCamara(ANCHO, i, cte, media, rmi);
            getContentPane().add(paneles[i]);
        }

        this.setBounds(20, 100, ANCHO, cte.getNumReceptores() * 103 + 53);
        this.setResizable(false);
        spnCamaras.addChangeListener(this);
        setControla(false);
        this.addWindowListener(this);
    }

    /**
     * Esta función es llamada cuando se le concede o se le quita al usuario
     * el control del vehículo. Se oculta el diálogo, se llama a la función
     * PnlCamara.setControla() y se modifica el tamaño del diálogo.
     * Una vez hecho esto, se vuelve a mostrar el diálogo.
     * @param activo Indica si se le concede o se le quita el control del vehículo
     */
    public void setControla(boolean activo) {
        setVisible(false);
        lblCamaras.setVisible(activo);
        spnCamaras.setVisible(activo);

        for (int i = 0; i < cte.getNumReceptores(); i++) {
            paneles[i].setControla(activo);
        }

        if (! activo) {
            setSize(ANCHO, cte.getNumReceptores() * 103 + 29);
        } else {
            setSize(ANCHO, cte.getNumReceptores() * 103 + 56);
        }
        setVisible(true);
    }

    /**
     * Setter para la propiedad <i>menu</i>
     * @param menu Nuevo valor para <i>menu</i>
     */
    public void setMenu(Menu menu) {
        this.menu = menu;
    }

    /**
     * Sobrecarga del método <i>toFront()</i> del diálogo. Hace que cuando se
     * llame a este método, se propague la llamada a cada uno de los paneles para
     * que se encarguen a su vez de traer al frente a cada uno de los visores del
     * stream multimedia
     */
    public synchronized void toFront() {
        super.toFront();
        for (int i = 0; i < paneles.length; i++) {
            paneles[i].toFront();
        }
    }

    /**
     * Evento <i>stateChanged</i>. Detecta si se ha pulsado el <i>spinners</i>
     * y llama a la función correspondiente del objeto RMI para que modifique el
     * ángulo en altura de las cámaras del vehículo en la parte del servidor
     * @param e ChangeEvent
     */
    public void stateChanged(ChangeEvent e) {
        // Comprueba que la fuente del evento es el spinner de las cámaras
        if (e.getSource() == spnCamaras) {
            float ang = ((Double)spnCamaras.getValue()).floatValue();
            // Transforma el valor para que quede entre 0 y 255
            ang = 255 * (90 - ang) / 180;
            try {
                // Hace la llamada al objeto RMI
                rmi.setAlturaCamaras((int)ang);
            } catch(RemoteException re) {
                Constantes.mensaje("No se pudo mover la cámara ", re);
            }
        }
    }

    /**
     * Evento <i>windowActivated</i>
     * @param e WindowEvent
     */
    public void windowActivated(WindowEvent e) {}
    /**
     * Evento <i>windowClosed</i>
     * @param e WindowEvent
     */
    public void windowClosed(WindowEvent e) {}
    /**
     * Evento <i>windowClosing</i>. Indica al objeto menu que la ventana se ha
     * cerrado, para que aparezca como desactivado.
     * @param e WindowEvent
     */
    public void windowClosing(WindowEvent e) {
        menu.setActivoCamaras(false);
    }
    /**
     * Evento <i>windowDeactivated</i>
     * @param e WindowEvent
     */
    public void windowDeactivated(WindowEvent e) {}
    /**
     * Evento <i>windowDeiconified</i>
     * @param e WindowEvent
     */
    public void windowDeiconified(WindowEvent e) {}
    /**
     * Evento <i>windowIconified</i>
     * @param e WindowEvent
     */
    public void windowIconified(WindowEvent e) {}
    /**
     * Evento <i>windowOpened</i>
     * @param e WindowEvent
     */
    public void windowOpened(WindowEvent e) {}

}
