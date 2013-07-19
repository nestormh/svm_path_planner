/**
 * Contiene las clases correspondiente a la interfaz de usuario que permite
 * configurar las opciones
 */
package carrito.cliente.interfaz.opciones;

import java.awt.*;
import java.awt.event.*;

import javax.swing.*;
import javax.swing.filechooser.*;

import carrito.cliente.interfaz.*;
import carrito.configura.*;
import carrito.media.*;
import org.jdesktop.layout.*;
import org.jdesktop.layout.GroupLayout;
import org.jdesktop.layout.LayoutStyle;

/**
 * Clase que muestra al usuario una interfaz gráfica que le permite configurar
 * los parámetros necesarios para efectuar la grabación del vídeo que llega
 * desde la aplicación servidor
 * @author Néstor Morales Hernández
 * @version 1.0
 */
public class DlgConfiguraRec extends JDialog implements MouseListener {

    // Variables de la interfaz
    private JButton btnAceptar;
    private JButton btnCancelar;
    private JButton btnFile;
    private JComboBox cmbBitrate;
    private JComboBox cmbCodec;
    private JComboBox cmbMux;
    private JComboBox cmbScale;
    private JLabel jLabel1;
    private JLabel lblBitrate;
    private JLabel lblMux;
    private JLabel lblScale;
    private JPanel pnlCompresion;
    private JLabel txtCodec;
    private JTextField txtFile;

    // Variables relacionadas con las extensiones del filtro y la traducción de
    // los datos seleccionados de forma que puedan ser entendidos por VLC
    private String[] mux = null;
    private String[][] extensiones = null;
    private String[] descripciones = null;

    // Variables a modificar
    private String ip = "";
    private int port = -1;
    private int caching = -1;

    /** Panel que controla la cámara seleccionada */
    private PnlCamara cam = null;

    /** Objeto multimedia */
    private ClienteMedia media = null;

    /**
     * Inicializa los parámetros necesarios para crear la interfaz. Una vez
     * inicializados, llama al método {@link #initComponents()}
     *
     * @param ip Dirección IP del servidor multimedia remoto
     * @param port Puerto del servidor multimedia remoto
     * @param caching Tamaño del buffer caché de las imágenes que llegan desde
     * el servidor
     * @param media Objeto de la clase ClienteMedia que realiza todas las
     * operaciones multimedia en la aplicación
     * @param cam Panel de control de la cámara que llamó al diálogo de
     * configuración de la grabación de video
     */
    public DlgConfiguraRec(String ip, int port, int caching, ClienteMedia media, PnlCamara cam) {
        super(new JDialog(), true);
        this.ip = ip;
        this.port = port;
        this.caching = caching;
        this.cam = cam;               // Identificador del panel de control de la cámara
        this.media = media;           // Objeto de control multimedia

        // Tipos de multiplexado (El nombre "no amigable" que se va a incluir en el comando final)
        mux = new String[] {"ts", "ps", "mpeg1", "mp4", "mov", "raw"};
        String[] mpg = {".mpg", ".mpeg"};
        String[] avi = {".avi"};
        String[] wmv = {".wmv"};
        // Extensiones asociadas a cada tipo de codec, empleadas por el filtro
        extensiones = new String[][] {mpg, mpg, mpg, avi, avi, avi, avi, avi,
                      wmv, wmv, mpg, avi};
        // Descripciones asociadas a cada extensión
        descripciones = new String[] {
                        "Archivo de vídeo MPEG-1 (*.mpg, *mpeg)",
                        "Archivo de vídeo MPEG-2 (*.mpg, *mpeg)",
                        "Archivo de vídeo MPEG-4 (*.mpg, *mpeg)",
                        "Archivo de vídeo DivX v1.0 (*.avi)",
                        "Archivo de vídeo DivX v3.0 (*.avi)",
                        "Archivo de vídeo H.263 (*.avi)",
                        "Archivo de vídeo H.264 (*.avi)",
                        "Archivo de Windows Media Video v1.0 (*.wmv)",
                        "Archivo de Windows Media Video v2.0 (*.wmv)",
                        "Archivo de vídeo MJPG (*.mpg, *mpeg)",
                        "Archivo de vídeo Theodora (*.avi)",
        };
        initComponents();
    }

    /**
     * Se encarga de inicializar la interfaz gráfica de la ventana
     */
    private void initComponents() {
        txtFile = new JTextField();
        btnFile = new JButton();
        btnCancelar = new JButton();
        btnAceptar = new JButton();
        jLabel1 = new JLabel();
        pnlCompresion = new JPanel();
        txtCodec = new JLabel();
        lblBitrate = new JLabel();
        lblScale = new JLabel();
        lblMux = new JLabel();
        cmbMux = new JComboBox();
        cmbCodec = new JComboBox();
        cmbBitrate = new JComboBox();
        cmbScale = new JComboBox();

        setResizable(false);
        setTitle("Configuración de la grabación");
        setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);

        btnFile.setText("...");

        btnCancelar.setText("Cancelar");

        btnAceptar.setText("Aceptar");

        jLabel1.setText("Fichero destino:");

        pnlCompresion.setBorder(BorderFactory.createTitledBorder(
                "Propiedades de compresi\u00f3n"));
        txtCodec.setText("Codec:");

        lblBitrate.setText("Bitrate:");

        lblScale.setText("Escala:");

        lblMux.setText("Multiplexado:");

        cmbMux.setModel(new DefaultComboBoxModel(new String[] {"MPEG TS", "MPEG PS", "MPEG 1", "MP4", "MOV", "RAW"}));
        cmbMux.setSelectedIndex(0);

        cmbCodec.setModel(new DefaultComboBoxModel(new String[] {"mp1v", "mp2v",
                                                   "mp4v", "DIV1", "DIV3", "H263", "H264", "WMV1", "WMV2", "MJPG",
                                                   "theo"}));
        cmbCodec.setSelectedIndex(4);

        cmbBitrate.setModel(new DefaultComboBoxModel(new String[] {"3072",
                                                     "2048", "1024", "768", "512", "384", "256", "192", "128", "96",
                                                     "64", "32", "16"}));
        cmbBitrate.setSelectedIndex(2);

        cmbScale.setModel(new DefaultComboBoxModel(new String[] {"0.25", "0.5",
                                                   "0.75", "1", "1.25", "1.5", "1.75", "2"}));
        cmbScale.setSelectedIndex(3);

        GroupLayout pnlCompresionLayout = new GroupLayout(pnlCompresion);
        pnlCompresion.setLayout(pnlCompresionLayout);
        pnlCompresionLayout.setHorizontalGroup(
                pnlCompresionLayout.createParallelGroup(GroupLayout.LEADING)
                .add(pnlCompresionLayout.createSequentialGroup()
                     .addContainerGap()
                     .add(pnlCompresionLayout.createParallelGroup(GroupLayout.
                LEADING)
                          .add(lblMux)
                          .add(txtCodec)
                          .add(lblBitrate)
                          .add(lblScale))
                     .addPreferredGap(LayoutStyle.RELATED, 40, Short.MAX_VALUE)
                     .add(pnlCompresionLayout.createParallelGroup(GroupLayout.
                LEADING, false)
                          .add(cmbScale, 0, GroupLayout.DEFAULT_SIZE,
                               Short.MAX_VALUE)
                          .add(cmbBitrate, 0, GroupLayout.DEFAULT_SIZE,
                               Short.MAX_VALUE)
                          .add(cmbCodec, 0, GroupLayout.DEFAULT_SIZE,
                               Short.MAX_VALUE)
                          .add(cmbMux, 0, 112, Short.MAX_VALUE))
                     .addContainerGap())
                );
        pnlCompresionLayout.setVerticalGroup(
                pnlCompresionLayout.createParallelGroup(GroupLayout.LEADING)
                .add(pnlCompresionLayout.createSequentialGroup()
                     .add(pnlCompresionLayout.createParallelGroup(GroupLayout.
                BASELINE)
                          .add(lblMux)
                          .add(cmbMux, GroupLayout.PREFERRED_SIZE,
                               GroupLayout.DEFAULT_SIZE,
                               GroupLayout.PREFERRED_SIZE))
                     .add(17, 17, 17)
                     .add(pnlCompresionLayout.createParallelGroup(GroupLayout.
                BASELINE)
                          .add(txtCodec)
                          .add(cmbCodec, GroupLayout.PREFERRED_SIZE,
                               GroupLayout.DEFAULT_SIZE,
                               GroupLayout.PREFERRED_SIZE))
                     .add(16, 16, 16)
                     .add(pnlCompresionLayout.createParallelGroup(GroupLayout.
                BASELINE)
                          .add(lblBitrate)
                          .add(cmbBitrate, GroupLayout.PREFERRED_SIZE,
                               GroupLayout.DEFAULT_SIZE,
                               GroupLayout.PREFERRED_SIZE))
                     .add(15, 15, 15)
                     .add(pnlCompresionLayout.createParallelGroup(GroupLayout.
                BASELINE)
                          .add(lblScale)
                          .add(cmbScale, GroupLayout.PREFERRED_SIZE,
                               GroupLayout.DEFAULT_SIZE,
                               GroupLayout.PREFERRED_SIZE))
                     .addContainerGap(GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                );

        GroupLayout layout = new GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
                layout.createParallelGroup(GroupLayout.LEADING)
                .add(layout.createSequentialGroup()
                     .addContainerGap(23, Short.MAX_VALUE)
                     .add(layout.createParallelGroup(GroupLayout.LEADING)
                          .add(layout.createSequentialGroup()
                               .add(183, 183, 183)
                               .add(btnFile, GroupLayout.PREFERRED_SIZE, 47,
                                    GroupLayout.PREFERRED_SIZE))
                          .add(layout.createSequentialGroup()
                               .add(82, 82, 82)
                               .add(btnAceptar)
                               .addPreferredGap(LayoutStyle.RELATED)
                               .add(btnCancelar))
                          .add(txtFile, GroupLayout.PREFERRED_SIZE, 176,
                               GroupLayout.PREFERRED_SIZE))
                     .addContainerGap())
                .add(layout.createSequentialGroup()
                     .addContainerGap()
                     .add(pnlCompresion, GroupLayout.PREFERRED_SIZE,
                          GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE)
                     .addContainerGap(13, Short.MAX_VALUE))
                .add(layout.createSequentialGroup()
                     .addContainerGap()
                     .add(jLabel1)
                     .addContainerGap(184, Short.MAX_VALUE))
                );
        layout.setVerticalGroup(
                layout.createParallelGroup(GroupLayout.LEADING)
                .add(layout.createSequentialGroup()
                     .add(9, 9, 9)
                     .add(pnlCompresion, GroupLayout.PREFERRED_SIZE,
                          GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE)
                     .addPreferredGap(LayoutStyle.RELATED)
                     .add(jLabel1)
                     .addPreferredGap(LayoutStyle.RELATED)
                     .add(layout.createParallelGroup(GroupLayout.BASELINE)
                          .add(btnFile)
                          .add(txtFile, GroupLayout.PREFERRED_SIZE,
                               GroupLayout.DEFAULT_SIZE,
                               GroupLayout.PREFERRED_SIZE))
                     .add(19, 19, 19)
                     .add(layout.createParallelGroup(GroupLayout.BASELINE)
                          .add(btnCancelar)
                          .add(btnAceptar))
                     .addContainerGap())
                );

        pack();

        Dimension d = Toolkit.getDefaultToolkit().getScreenSize();
        setLocation((d.width - getWidth()) / 2, (d.height - getHeight()) / 2);

        btnFile.addMouseListener(this);
        btnCancelar.addMouseListener(this);
        btnAceptar.addMouseListener(this);
    }
    /**
     * Evento <i>mouseClicked</i>.
     * <p>Si se pulsa Aceptar, se crea una nueva instancia multimedia que
     * <i>escuchará</i> al servidor, transformará el vídeo recibido según la
     * configuración del cliente, y almacenará el resultado en el fichero indicado</p>
     * <p>Si se pulsa el botón <i>...</i>, se abrirá un cuadro de diálogo que
     * permitirá elegir el fichero en el que se desea guardar el resultado</p>
     * @param e MouseEvent
     */
    public void mouseClicked(MouseEvent e) {
        if (e.getButton() == MouseEvent.BUTTON1) {    // Se ha pulsado el botón izquierdo
            if (e.getSource() == btnFile) {           // Se pulsó el botón "..."
                if (btnFile.isEnabled()) {
                    int index = cmbCodec.getSelectedIndex();

                    FileSystemView fsv = FileSystemView.getFileSystemView();
                    JFileChooser fc = new JFileChooser(fsv.getHomeDirectory());

                    Filtro filtro = new Filtro(extensiones[index],
                                               descripciones[index]);
                    fc.addChoosableFileFilter(filtro);
                    fc.setFileFilter(filtro);

                    // Muestra el diálogo de selección del fichero
                    if (fc.showSaveDialog(this) == JFileChooser.APPROVE_OPTION) {
                        String fichero = fc.getSelectedFile().toString();
                        boolean conExt = false;
                        for (int i = 0; i < extensiones[index].length; i++) {
                            if (fichero.endsWith(extensiones[index][i])) {
                                conExt = true;
                                break;
                            }
                        }
                        if (!conExt) {
                            fichero += extensiones[cmbCodec.getSelectedIndex()][0];
                        }
                        // Establece el nombre de fichero en la interfaz
                        txtFile.setText(fichero);
                    }
                }
            } else if (e.getSource() == btnCancelar) {       // Se canceló el diálogo
                this.dispose();
            } else if (e.getSource() == btnAceptar) {        // Aceptar
                String codec = (String) cmbCodec.getSelectedItem();
                int bitrate = Integer.parseInt((String) cmbBitrate.
                                               getSelectedItem());
                float scale = Float.parseFloat((String) cmbScale.
                                               getSelectedItem());
                String muxName = mux[cmbMux.getSelectedIndex()];
                String file = (txtFile.getText().length() != 0) ?
                              txtFile.getText().replaceAll(" ", "&nbsp;") : null;

                try {
                    // Se crea una nueva instancia encargada de grabar en el fichero
                    int cliente = media.addCliente(ip, port, caching, codec, bitrate, scale, muxName, file, false);
                    cam.setMedia(cliente);
                } catch (Exception ex) {
                    Constantes.mensaje("No se pudo iniciar la grabación" +
                                    ex.getMessage());
                }

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
