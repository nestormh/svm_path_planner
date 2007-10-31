/**
 * Paquete que contiene las clases correspondientes a la interfaz del cliente
 */
package carrito.cliente.interfaz;

import java.awt.event.*;

import javax.swing.*;

import carrito.cliente.interfaz.opciones.*;
import carrito.configura.*;
import carrito.server.*;
import java.rmi.RemoteException;

/**
 * Clase que construye el menú de la aplicación.
 * @author Néstor Morales Hernández
 * @version 1.0
 */
public class Menu extends JMenuBar implements MouseListener {
    // Menús principales
    private JMenu mnuVentanas = new JMenu("Ventanas");
    private JMenu mnuOpciones = new JMenu("Opciones");

    // Submenús del menú ventanas
    private JCheckBoxMenuItem mnuVentanasCamaras = new JCheckBoxMenuItem("Ver control de las cámaras", null, true);
    private JMenuItem mnuVentanasJoy = new JMenuItem("Solicitar Joystick");

    // Submenús del menú de opciones
    private JMenuItem mnuOpcionesJoy = new JMenuItem("Calibrar joystick");
    private JMenuItem mnuOpcionesOtros = new JMenuItem("Otras opciones...");

    // Diálogos de la aplicación
    private DlgCamaras camaras = null;
    private DlgSolicitaJoy dsj = null;
    private frmCliente form = null;

    /** Objeto que hace de interfaz entre todas las variables comunes a la aplicación */
    private Constantes cte = null;
    /** Objeto RMI */
    private InterfazRMI rmi = null;

    /** Variable para controlar que al activar la aplicación no se produzca un ciclo
     * al traer el resto de ventanas al frente
     */
    private boolean activo = false;

    /**
     * Constructor de la clase Menu. Crea el menú principal de la aplicación
     * @param cte Objeto que hace de interfaz entre todas las variables comunes a la aplicación
     * @param form Ventana principal de la aplicación
     * @param rmi Objeto RMI
     * @param camaras Diálogo de control de las cámaras
     */
    public Menu(Constantes cte, frmCliente form, InterfazRMI rmi, DlgCamaras camaras) {
        this.cte = cte;
        this.form = form;
        this.camaras = camaras;
        this.rmi = rmi;

        this.add(mnuVentanas);
        this.add(mnuOpciones);

        mnuVentanas.add(mnuVentanasCamaras);
        try {
            if (rmi.isControlActivo())
                mnuVentanas.add(mnuVentanasJoy);
        } catch (RemoteException re) {}
        mnuOpciones.add(mnuOpcionesJoy);
        mnuOpciones.add(mnuOpcionesOtros);

        mnuVentanasJoy.addMouseListener(this);
        mnuVentanasCamaras.addMouseListener(this);
        mnuOpcionesJoy.addMouseListener(this);
        mnuOpcionesOtros.addMouseListener(this);
    }

    /**
     * Método que trae al frente todas las ventanas de la aplicación
     */
    public void toFront() {
        // Si es la primera vez que se llama al método, trae todas las ventanas al frente
        if (! activo) {
            camaras.toFront();
            if (dsj != null)
                dsj.toFront();
            activo = true;
            // Si no, cambia el valor de la variable activo para que no se produzcan ciclos
        } else {
            activo = false;
        }
    }

    /**
     * Método que indica si la ventana de control de las cámaras está visible o no
     * @param valor boolean
     */
    public void setActivoCamaras(boolean valor) {
        mnuVentanasCamaras.setState(valor);
    }

    /**
     * Evento <i>mouseClicked</i>
     * @param e MouseEvent
     */
    public void mouseClicked(MouseEvent e) {}
    /**
     * Evento <i>mouseEntered</i>
     * @param e MouseEvent
     */
    public void mouseEntered(MouseEvent e) {}
    /**
     * Evento <i>mouseExited</i>
     * @param e MouseEvent
     */
    public void mouseExited(MouseEvent e) {}
    /**
     * Evento <i>mousePressed</i>
     * @param e MouseEvent
     */
    public void mousePressed(MouseEvent e) {}

    /**
     * Evento <i>mouseReleased</i>. Comprueba cuál de los menús fue seleccionado
     * y realiza la acción correspondiente a cada uno
     * @param e MouseEvent
     */
    public void mouseReleased(MouseEvent e) {
        // Se pulsó el botón izquierdo
        if (e.getButton() == MouseEvent.BUTTON1) {
            // Si la fuente fue la solicitud del Joystick, crea una nueva ventana
            // de solicitud, siempre y cuando no exista ya una
            if (e.getSource() == mnuVentanasJoy) {
                if ((dsj == null) || (! dsj.isVisible())){
                    dsj = new DlgSolicitaJoy(cte, rmi, camaras);
                }
                // Si se seleccionó la ventana de control de las cámaras, esta
                // se muestra o se oculta según proceda
            } else if (e.getSource() == mnuVentanasCamaras) {
                if (mnuVentanasCamaras.getState()) {
                    camaras.setVisible(true);;
                } else {
                    camaras.setVisible(false);;
                }
                // Si se selecciona el asistente de calibrado, se crea un nuevo
                // asistente
            } else if (e.getSource() == mnuOpcionesJoy) {
                DlgOptJoystick doj = new DlgOptJoystick(form, cte);
                doj.setVisible(true);
                // Si se selecciona la configuración del cliente, se muestra la
                // ventana de configuración
            } else if (e.getSource() == this.mnuOpcionesOtros) {
                DlgConfiguraClt dcc = new DlgConfiguraClt(form, cte);
                dcc.setVisible(true);
            }
        }
    }
}

