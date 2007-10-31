/**
 * Paquete que contiene las clases correspondientes a la interfaz del cliente
 */
package carrito.cliente.interfaz;

import java.awt.*;
import java.awt.event.*;

import javax.swing.*;

import carrito.configura.*;
import carrito.server.*;

/**
 * Clase que crea la interfaz principal de la aplicación. Esta interfaz contiene
 * únicamente un Menú que permite acceder a los distintos diálogos que componen
 * la aplicación.
 * @author Néstor Morales Hernández
 * @version 1.0
 */
public class frmCliente extends JFrame {
    // Constantes que indican las dimensiones de la ventana
    private static final int ANCHO = 700;
    private static final int ALTO = 50;

    /** Objeto menú de la aplicación */
    private Menu menu = null;

    /**
     * Contructor que crea la interfaz y crea un nuevo objeto Menu
     * @param cte Objeto que hace de interfaz entre todas las variables
     * comunes a la aplicación
     * @param rmi Objeto RMI
     * @param camaras Diálogo de control de las cámaras
     */
    public frmCliente(Constantes cte, InterfazRMI rmi, DlgCamaras camaras) {
        setIconImage(new ImageIcon("cars.jpg").getImage());
        setResizable(false);
        setTitle("Monitor del Carrito de Golf");
        int scrX = Toolkit.getDefaultToolkit().getScreenSize().width;
        int x = (scrX / 2) - (ANCHO / 2);
        setBounds(x, 10, ANCHO, ALTO);
        getContentPane().setLayout(new FlowLayout());
        menu = new Menu(cte, this, rmi, camaras);
        this.setJMenuBar(menu);

        // Añade un listener
        addWindowListener( new WindowAdapter() {
            // Si se activa la ventana, se activan los demás diálogos (Esto se hace
            // para que en caso de que el usuario esté ejecutando varias aplicaciones,
            // cuando se active la aplicación principal, que venga al frente)
            public void windowActivated (WindowEvent ev) {
                menu.toFront();
                toFront();
            }

            public void windowClosing(WindowEvent ev) {
                System.exit(0);
            }
        } );
        camaras.setMenu(menu);
        camaras.setVisible(true);
    }

    /**
     * Getter del objeto menu
     * @return Menu - Propiedad menú de la clase
     */
    public Menu getMenu() {
        return menu;
    }
}
