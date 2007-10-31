/**
 * Contiene las clases correspondiente a la interfaz de usuario que permite
 * configurar las opciones
 */
package carrito.cliente.interfaz.opciones;

import java.awt.*;
import java.awt.event.*;
import java.util.regex.*;

import javax.swing.*;

import carrito.configura.*;
import org.jdesktop.layout.*;
import org.jdesktop.layout.GroupLayout;
import org.jdesktop.layout.LayoutStyle;

/**
 * Clase encargada de mostrar al usuario la interfaz  gráfica por medio de
 * la cual se va a proceder a configurar los parámetros de conexión del cliente
 * con el servidor, además de aquellos refentes a la propia aplicación local
 *
 * @author Néstor Morales Hernández
 * @version 1.0
 */

public class DlgConfiguraClt extends JDialog implements MouseListener {
    // Variables de la interfaz
    private JButton btnAceptar;
    private JButton btnCancelar;
    private JButton btnJoy;
    private JLabel lblCaching;
    private JLabel lblIP;
    private JPanel pnlVideo;
    private JSpinner spnCaching;
    private JTextField txtIP;

    /** Objeto que hace de interfaz de todas las variables comunes de la aplicación */
    private Constantes cte = null;

    /**
     * Constructor de la clase. Asigna el <i>owner</i> y la clase encargada del
     * manejo de los parámetros generales. Cuando asigna estos valores llama al
     * método privado {@link  #initComponents()}
     * @param owner Se trata del <i>owner</i> de la ventana
     * @param cte Constantes Clase encargada de la gestión de las variables en la aplicación
     */
    public DlgConfiguraClt(JFrame owner, Constantes cte) {
        super(owner, true);
        this.cte = cte;
        initComponents();
    }

    /**
     * Se encarga de inicializar la interfaz gráfica de la ventana
     */
    private void initComponents() {
        pnlVideo = new JPanel();
        lblIP = new JLabel();
        txtIP = new JTextField();
        spnCaching = new JSpinner(new SpinnerNumberModel(cte.getCltCaching(), 1, 10000, 10));
        lblCaching = new JLabel();
        btnJoy = new JButton();
        btnAceptar = new JButton();
        btnCancelar = new JButton();

        setTitle("Configuración");
        setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
        pnlVideo.setBorder(BorderFactory.createTitledBorder("Configuraci\u00f3n de V\u00eddeo"));
        lblIP.setText("Direcci\u00f3n IP del Servidor:");

        txtIP.setText(cte.getIp());

        lblCaching.setText("Cach\u00e9");

        org.jdesktop.layout.GroupLayout pnlVideoLayout = new org.jdesktop.layout.GroupLayout(pnlVideo);
        pnlVideo.setLayout(pnlVideoLayout);
        pnlVideoLayout.setHorizontalGroup(
            pnlVideoLayout.createParallelGroup(org.jdesktop.layout.GroupLayout.LEADING)
            .add(pnlVideoLayout.createSequentialGroup()
                .addContainerGap()
                .add(pnlVideoLayout.createParallelGroup(org.jdesktop.layout.GroupLayout.LEADING)
                    .add(lblIP)
                    .add(txtIP, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE, 141, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE)
                    .add(lblCaching)
                    .add(spnCaching, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE, 62, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE))
                .addContainerGap(org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );
        pnlVideoLayout.setVerticalGroup(
            pnlVideoLayout.createParallelGroup(org.jdesktop.layout.GroupLayout.LEADING)
            .add(pnlVideoLayout.createSequentialGroup()
                .add(lblIP)
                .addPreferredGap(org.jdesktop.layout.LayoutStyle.RELATED)
                .add(txtIP, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE)
                .add(7, 7, 7)
                .add(lblCaching)
                .addPreferredGap(org.jdesktop.layout.LayoutStyle.RELATED)
                .add(spnCaching, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE)
                .addContainerGap(org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );

        btnJoy.setText("Calibrar Joystick");

        btnAceptar.setText("Aceptar");

        btnCancelar.setText("Cancelar");

        GroupLayout layout = new GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(GroupLayout.LEADING)
            .add(layout.createSequentialGroup()
                .add(layout.createParallelGroup(GroupLayout.LEADING)
                    .add(layout.createSequentialGroup()
                        .addContainerGap()
                        .add(pnlVideo, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
                    .add(layout.createSequentialGroup()
                        .add(44, 44, 44)
                        .add(btnJoy))
                    .add(layout.createSequentialGroup()
                        .add(18, 18, 18)
                        .add(btnAceptar)
                        .addPreferredGap(LayoutStyle.RELATED)
                        .add(btnCancelar)))
                .addContainerGap(GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(GroupLayout.LEADING)
            .add(layout.createSequentialGroup()
                .addContainerGap()
                .add(pnlVideo, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(LayoutStyle.RELATED)
                .add(btnJoy)
                .add(16, 16, 16)
                .add(layout.createParallelGroup(GroupLayout.BASELINE)
                    .add(btnCancelar)
                    .add(btnAceptar))
                .addContainerGap(GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );
        pack();

        Dimension d = Toolkit.getDefaultToolkit().getScreenSize();
        setLocation((d.width - getWidth()) / 2, (d.height - getHeight()) / 2);

        btnJoy.addMouseListener(this);
        btnAceptar.addMouseListener(this);
        btnCancelar.addMouseListener(this);
    }

    /**
     * Evento <i>mouseClicked</i>. Si se pulsa Aceptar,
     * las opciones son almacenadas en la clase Constantes, la cual a su vez se encarga
     * de guardar las opciones en un fichero para poder recuperar estos valores la próxima
     * vez que se inicie la aplicación
     * @param e MouseEvent
     */
    public void mouseClicked(MouseEvent e) {
        if (e.getButton() == MouseEvent.BUTTON1) {
            if (e.getSource() == btnJoy) { // Se seleccionó el calibrado del Joystick
                DlgOptJoystick doj = new DlgOptJoystick(this, cte);
                doj.show();
            } else if(e.getSource() == btnCancelar) {   // Cancelando
                dispose();
            } else if (e.getSource() == btnAceptar) {   // Aceptar. Se guardan las opciones
                String ip = txtIP.getText();
                int caching = ((Integer)spnCaching.getValue()).intValue();

                if ((caching < 1) || (caching > 10000)) {
                    Constantes.mensaje("El tamaño de caché " + caching + " es inválido");
                    return;
                }

                String ip2 = ip.replaceAll("\\p{Punct}", "p");
                Pattern p = Pattern.compile("\\d{1,3}p\\d{1,3}p\\d{1,3}p\\d{1,3}");
                Matcher m = p.matcher(ip2);
                if (! m.matches()) {
                    Constantes.mensaje("La IP " + ip + " no es correcta");
                    return;
                }

                cte.setCltCaching(caching);
                cte.setIp(ip);
                cte.saveCliente();
                dispose();
            }
        }
    }
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
         * Evento <i>mouseReleased</i>
         * @param e MouseEvent
         */
    public void mouseReleased(MouseEvent e) {}

}
