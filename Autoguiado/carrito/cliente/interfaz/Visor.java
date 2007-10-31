/**
 * Paquete que contiene las clases correspondientes a la interfaz del cliente
 */
package carrito.cliente.interfaz;

import java.awt.*;

import javax.swing.*;

import carrito.media.*;

/**
 * Clase que crea un diálogo que contiene un panel encargado de visualizar el contenido multimedia
 * @author Néstor Morales Hernández
 * @version 1.0
 */
public class Visor extends JDialog {
    /** Panel que visualiza el contenido multimedia */
    private MediaCanvas panel = null;
    /** Objeto multimedia */
    private Media media = null;
    /** Identificador de la instancia de la reproducción en VLC */
    private int id = -1;

    /**
     * Constructor que crea la interfaz del diálogo
     * @param icono Frame padre del cual se va a heredar el icono
     * @param media Objeto multimedia común a toda la aplicación
     * @param id Identificador de la instancia VLC
     * @param posX Posición horizontal del cuadro de diálogo dentro de la pantalla
     * @param posY Posición vertical del cuadro de diálogo dentro de la pantalla
     * @param width Ancho de la ventana
     * @param height Alto de la ventana
     */
    public Visor(JFrame icono, Media media, int id, int posX, int posY, int width, int height) {
        super(icono);
        this.media = media;
        this.id = id;
        panel = new MediaCanvas(id);
        panel.paint(null, 0);
        this.setDefaultCloseOperation(DO_NOTHING_ON_CLOSE);
        this.getContentPane().setLayout(new FlowLayout());
        panel.setSize(new Dimension(width, height));
        getContentPane().add(panel);
        setLocation(posX, posY);
        pack();
        super.setVisible(true);
        setVisible(true);
        media.play(id);
    }

    /**
     * Sobrecarga del método paint que permite actualizar el tamaño del panel
     * cuando se modifica el tamaño de la ventana
     * @param g Graphics
     */
    public void paint(Graphics g) {
        super.paint(g);
        panel.setBounds(0,0,this.getWidth() - 8, this.getHeight() - 27);
    }

    /**
     * Inicia la reproducción
     * @return Devuelve false si hubo algún problema al iniciar la reproducción
     */
    public boolean play() {
        if (! isVisible())
            setVisible(true);
        return media.play(id);
    }

    /**
     * Pausa la reproducción
     * @return Devuelve false si hubo algún problema al pausar la reproducción
     */
    public boolean pausa() {
        return media.pausa(id);
    }
    /**
     * Detiene la reproducción y oculta la ventana
     * @return Devuelve false si hubo algún problema al detener la reproducción
     */
    public boolean stop() {
        setVisible(false);
        return media.stop(id);
    }
}
