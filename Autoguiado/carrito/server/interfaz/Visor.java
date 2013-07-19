/**
 * Paquete que contiene las clases correspondientes a la interfaz de la aplicación
 * servidor
 */
package carrito.server.interfaz;

import java.awt.*;
import java.awt.event.*;

import javax.swing.*;

import carrito.configura.*;
import carrito.media.*;

/**
 * Clase que crea un diálogo que contiene un panel encargado de visualizar el contenido multimedia
 * @author Néstor Morales Hernández
 * @version 1.0
 */
public class Visor extends JDialog implements WindowListener {
    /** Panel que visualiza el contenido multimedia */
    private MediaCanvas panel = null;

    /**
     * Constructor que crea la interfaz del diálogo
     * @param icono Frame padre del cual se va a heredar el icono
     * @param media Objeto multimedia común a toda la aplicación
     * @param id Identificador de la instancia VLC
     * @param vsc VideoSvrConf
     */
    public Visor(JFrame icono, Media media, int id, VideoSvrConf vsc) {
        super(icono);
        setTitle("Cámara " + (vsc.getVideoDisp() + 1));
        panel = new MediaCanvas(id);
        panel.paint(null, 0);
        this.getContentPane().setLayout(new FlowLayout());
        panel.setSize(new Dimension(vsc.getWidth(), vsc.getHeight()));
        add(panel);
        pack();
        super.setVisible(true);
        setVisible(true);
        media.play(id);
        addWindowListener(this);
        this.setDefaultCloseOperation(DO_NOTHING_ON_CLOSE);
    }

    /**
     * Sobrecarga del método paint que permite actualizar el tamaño del panel
     * cuando se modifica el tamaño de la ventana
     * @param g Graphics
     */
    public void paint(Graphics g) {
        super.paint(g);
        panel.setBounds(0, 0, this.getWidth() - 8, this.getHeight() - 27);
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
     * Evento <i>windowClosing</i>. Pide confirmación al usuario de que desea
     * cerrar la ventana, ya que una vez cerrada no será posible abrirla de nuevo
     * @param e WindowEvent
     */
    public void windowClosing(WindowEvent e) {
        if (JOptionPane.showConfirmDialog(this,
                "¿Está seguro que desea cerrar la ventana?\rNo podrá volver " +
                                          "a abrirla a menos que reinicie la aplicación",
                                          "¿Desea cerrar la ventana?",
                                          JOptionPane.YES_NO_OPTION,
                                          JOptionPane.QUESTION_MESSAGE) ==
            JOptionPane.YES_OPTION) {
            setVisible(false);
            dispose();
        }
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
