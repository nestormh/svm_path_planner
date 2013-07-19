/**
 * Paquete que contiene las clases correspondientes a la interfaz de las opciones
 * del servidor
 */
package carrito.server.interfaz.opciones;

import java.awt.*;
import java.awt.event.*;
import java.util.*;
import java.util.regex.*;

import javax.comm.*;
import javax.swing.*;
import javax.swing.filechooser.*;

import carrito.configura.*;
import org.jdesktop.layout.*;
import org.jdesktop.layout.GroupLayout;
import org.jdesktop.layout.LayoutStyle;

/**
 * Clase que crea una interfaz para llevar a cabo la configuración de cada uno de
 * los dispositivos
 * @author Néstor Morales Hernández
 * @version 1.0
 */
public class DlgOpcionesCamaras extends JDialog implements MouseListener {

    /** Lista de tipos de multiplexado */
    private String[] mux = null;
    /** Extensiones asociadas a cada tipo de multiplexado */
    private String[][] extensiones = null;
    /** Descripción de cada tipo de multiplexado */
    private String[] descripciones = null;

    // Variables de la interfaz
    private JButton btnAceptar;
    private JButton btnCancelar;
    private JButton btnFile;
    private JCheckBox cbFile;
    private JCheckBox cbVisible;
    private JComboBox cmbBitrate;
    private JComboBox cmbCodec;
    private JComboBox cmbMux;
    private JComboBox cmbPuerto;
    private JComboBox cmbScale;
    private JLabel lblBitrate;
    private JLabel lblCaching;
    private JLabel lblFps;
    private JLabel lblIP;
    private JLabel lblMux;
    private JLabel lblPort;
    private JLabel lblPuerto;
    private JLabel lblScale;
    private JLabel lblTamano;
    private JLabel lblX;
    private JPanel pnlCaptura;
    private JPanel pnlCompresion;
    private JSpinner spnCaching;
    private JSpinner spnFps;
    private JLabel txtCodec;
    private JTextField txtFile;
    private JTextField txtHeight;
    private JTextField txtIP;
    private JTextField txtPort;
    private JTextField txtWidth;

    /** Descriptor de la configuración de cada dispositivo de captura */
    private VideoSvrConf vsc = null;
    /** Lista de puertos COM activos */
    private String[] puertos = null;

    /**
     * Constructor. Inicializa las variables, obtiene la lista de puertos COM
     * activos e inicializa la interfaz
     * @param owner Ventana que llama a este diálogo (Ventana de configuración del
     * servidor
     * @param vsc Clase descriptora del servidor multimedia asociado al dispositivo
     */
    public DlgOpcionesCamaras(JFrame owner, VideoSvrConf vsc) {
        super(owner, true);
        this.vsc = vsc;

        // Crea la lista de tipos de multiplexado, extensiones asociadas y descripciones
        mux = new String[]{"ts", "ps", "mpeg1", "mp4", "mov", "raw"};
        String[] mpg = { ".mpg", ".mpeg" };
        String[] avi = { ".avi" };
        String[] wmv = { ".wmv" };
        extensiones = new String[][]{ mpg, mpg, mpg, avi, avi, avi, avi, avi, wmv, wmv, mpg, avi };
        descripciones = new String[]{
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

        // Crea la lista de puertos COM activos
        Enumeration ports = CommPortIdentifier.getPortIdentifiers();
        Vector tmp = new Vector();
        CommPortIdentifier port = null;
        while ((port = (CommPortIdentifier) ports.nextElement()) != null) {
            if (port.getPortType() == CommPortIdentifier.PORT_SERIAL) {
                tmp.add(port.getName());
            }
        }
        puertos = new String[tmp.size()];
        for (int i = 0; i < tmp.size(); i++) {
            puertos[i] = (String)tmp.elementAt(i);
        }

        // Crea la interfaz
        initComponents();
    }

    /**
     * Inicializa la interfaz
     */
    private void initComponents() {
        pnlCaptura = new JPanel();
        lblTamano = new JLabel();
        txtWidth = new JTextField();
        lblX = new JLabel();
        txtHeight = new JTextField();
        lblCaching = new JLabel();
        spnCaching = new JSpinner(new SpinnerNumberModel(vsc.getCaching(), 0, 10000, 10));
        lblFps = new JLabel();
        spnFps = new JSpinner(new SpinnerNumberModel(vsc.getFps(), 0.0f, 100.0f, 0.5f));
        lblPuerto = new javax.swing.JLabel();
        cmbPuerto = new javax.swing.JComboBox();
        pnlCompresion = new JPanel();
        txtCodec = new JLabel();
        lblBitrate = new JLabel();
        lblScale = new JLabel();
        lblMux = new JLabel();
        cmbMux = new JComboBox();
        cmbCodec = new JComboBox();
        cmbBitrate = new JComboBox();
        cmbScale = new JComboBox();
        cbVisible = new JCheckBox();
        lblIP = new JLabel();
        txtIP = new JTextField();
        lblPort = new JLabel();
        txtPort = new JTextField();
        cbFile = new JCheckBox();
        txtFile = new JTextField();
        btnFile = new JButton();
        btnAceptar = new JButton();
        btnCancelar = new JButton();

        setResizable(false);
        setTitle(vsc.getName());
        setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
        pnlCaptura.setBorder(BorderFactory.createTitledBorder("Propiedades de Captura"));
        lblTamano.setText("Tama\u00f1o:");

        txtWidth.setText(String.valueOf(vsc.getWidth()));

        lblX.setText("x");

        txtHeight.setText(String.valueOf(vsc.getHeight()));

        lblCaching.setText("Cach\u00e9: ");

        lblFps.setText("fps:");

        lblPuerto.setText(" Puerto de zoom:");

        cmbPuerto.setModel(new DefaultComboBoxModel(puertos));
        for (int i = 0; i < puertos.length; i++) {
            if (puertos[i].equals(vsc.getSerial())) {
                cmbPuerto.setSelectedIndex(i);
                break;
            }
        }

        GroupLayout pnlCapturaLayout = new GroupLayout(pnlCaptura);
                pnlCaptura.setLayout(pnlCapturaLayout);
                pnlCapturaLayout.setHorizontalGroup(
                    pnlCapturaLayout.createParallelGroup(GroupLayout.LEADING)
                    .add(pnlCapturaLayout.createSequentialGroup()
                        .addContainerGap()
                        .add(pnlCapturaLayout.createParallelGroup(GroupLayout.LEADING)
                            .add(lblTamano)
                            .add(lblCaching)
                            .add(lblFps)
                            .add(lblPuerto))
                        .addPreferredGap(LayoutStyle.RELATED, 111, Short.MAX_VALUE)
                        .add(pnlCapturaLayout.createParallelGroup(GroupLayout.LEADING)
                            .add(cmbPuerto, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE)
                            .add(pnlCapturaLayout.createSequentialGroup()
                                .add(txtWidth, GroupLayout.PREFERRED_SIZE, 30, GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(LayoutStyle.RELATED)
                                .add(lblX)
                                .addPreferredGap(LayoutStyle.RELATED)
                                .add(txtHeight, GroupLayout.PREFERRED_SIZE, 30, GroupLayout.PREFERRED_SIZE))
                            .add(spnFps, GroupLayout.PREFERRED_SIZE, 37, GroupLayout.PREFERRED_SIZE)
                            .add(spnCaching, GroupLayout.PREFERRED_SIZE, 51, GroupLayout.PREFERRED_SIZE))
                        .addContainerGap())
                );
                pnlCapturaLayout.setVerticalGroup(
                    pnlCapturaLayout.createParallelGroup(GroupLayout.LEADING)
                    .add(pnlCapturaLayout.createSequentialGroup()
                        .add(pnlCapturaLayout.createParallelGroup(GroupLayout.BASELINE)
                            .add(lblTamano)
                            .add(txtWidth, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE)
                            .add(lblX)
                            .add(txtHeight, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
                        .addPreferredGap(LayoutStyle.RELATED)
                        .add(pnlCapturaLayout.createParallelGroup(GroupLayout.BASELINE)
                            .add(lblCaching)
                            .add(spnCaching, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
                        .addPreferredGap(LayoutStyle.RELATED)
                        .add(pnlCapturaLayout.createParallelGroup(GroupLayout.BASELINE)
                            .add(lblFps)
                            .add(spnFps, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
                        .addPreferredGap(LayoutStyle.RELATED, GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                        .add(pnlCapturaLayout.createParallelGroup(GroupLayout.BASELINE)
                            .add(lblPuerto)
                            .add(cmbPuerto, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE)))
                );

        pnlCompresion.setBorder(BorderFactory.createTitledBorder("Propiedades de compresi\u00f3n"));
        txtCodec.setText("Codec:");

        lblBitrate.setText("Bitrate:");

        lblScale.setText("Escala:");

        lblMux.setText("Multiplexado:");

        cmbMux.setModel(new DefaultComboBoxModel(new String[] { "MPEG TS", "MPEG PS", "MPEG 1", "MP4", "MOV", "RAW" }));
        for (int i = 0; i < mux.length; i++) {
            if (vsc.getMux().equals(mux[i])) {
                cmbMux.setSelectedIndex(i);
                break;
            }
        }

        cmbCodec.setModel(new DefaultComboBoxModel(new String[] { "mp1v", "mp2v", "mp4v", "DIV1", "DIV3", "H263", "H264", "WMV1", "WMV2", "MJPG", "theo" }));
        for (int i = 0; i < cmbCodec.getModel().getSize(); i++) {
            if (vsc.getCodec().equals((String)cmbCodec.getModel().getElementAt(i))) {
                cmbCodec.setSelectedIndex(i);
                break;
            }
        }

        cmbBitrate.setModel(new DefaultComboBoxModel(new String[] { "3072", "2048", "1024", "768", "512", "384", "256", "192", "128", "96", "64", "32", "16" }));
        for (int i = 0; i < cmbBitrate.getModel().getSize(); i++) {
            if (vsc.getBitrate() == Integer.parseInt((String)cmbBitrate.getModel().getElementAt(i))) {
                cmbBitrate.setSelectedIndex(i);
                break;
            }
        }
        cmbBitrate.setEditable(true);

        cmbScale.setModel(new DefaultComboBoxModel(new String[] { "0.25", "0.5", "0.75", "1", "1.25", "1.5", "1.75", "2" }));
        for (int i = 0; i < cmbScale.getModel().getSize(); i++) {
            if (vsc.getScale() == Float.parseFloat((String)cmbScale.getModel().getElementAt(i))) {
                cmbScale.setSelectedIndex(i);
                break;
            }
        }
        cmbScale.setEditable(true);

        GroupLayout pnlCompresionLayout = new GroupLayout(pnlCompresion);
        pnlCompresion.setLayout(pnlCompresionLayout);
        pnlCompresionLayout.setHorizontalGroup(
            pnlCompresionLayout.createParallelGroup(GroupLayout.LEADING)
            .add(pnlCompresionLayout.createSequentialGroup()
                .addContainerGap()
                .add(pnlCompresionLayout.createParallelGroup(GroupLayout.LEADING)
                    .add(lblMux)
                    .add(txtCodec)
                    .add(lblBitrate)
                    .add(lblScale))
                .addPreferredGap(LayoutStyle.RELATED, 89, Short.MAX_VALUE)
                .add(pnlCompresionLayout.createParallelGroup(GroupLayout.LEADING, false)
                    .add(cmbScale, 0, GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .add(cmbBitrate, 0, GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .add(cmbCodec, 0, GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .add(cmbMux, GroupLayout.PREFERRED_SIZE, 112, GroupLayout.PREFERRED_SIZE))
                .addContainerGap())
        );
        pnlCompresionLayout.setVerticalGroup(
            pnlCompresionLayout.createParallelGroup(GroupLayout.LEADING)
            .add(pnlCompresionLayout.createSequentialGroup()
                .add(pnlCompresionLayout.createParallelGroup(GroupLayout.BASELINE)
                    .add(lblMux)
                    .add(cmbMux, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
                .add(17, 17, 17)
                .add(pnlCompresionLayout.createParallelGroup(GroupLayout.BASELINE)
                    .add(txtCodec)
                    .add(cmbCodec, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
                .add(16, 16, 16)
                .add(pnlCompresionLayout.createParallelGroup(GroupLayout.BASELINE)
                    .add(lblBitrate)
                    .add(cmbBitrate, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
                .add(15, 15, 15)
                .add(pnlCompresionLayout.createParallelGroup(GroupLayout.BASELINE)
                    .add(lblScale)
                    .add(cmbScale, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
                .addContainerGap(GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );

        cbVisible.setText("Previsualizar el v\u00eddeo capturado");
        cbVisible.setBorder(BorderFactory.createEmptyBorder(0, 0, 0, 0));
        cbVisible.setMargin(new java.awt.Insets(0, 0, 0, 0));
        cbVisible.setSelected(vsc.isDisplay());

        lblIP.setText("Direcci\u00f3n IP");
        txtIP.setText(vsc.getIp());
        lblPort.setText("Puerto");

        txtPort.setText(Integer.toString(vsc.getPort()));

        cbFile.setText("Escribir v\u00eddeo capturado a fichero");
        cbFile.setBorder(BorderFactory.createEmptyBorder(0, 0, 0, 0));
        cbFile.setMargin(new java.awt.Insets(0, 0, 0, 0));
        cbFile.setSelected(((vsc.getFile() == null) || (vsc.getFile().length() == 0))? false:true);

        txtFile.setText(vsc.getFile());
        txtFile.setEnabled(cbFile.isSelected());

        btnFile.setText("...");
        btnFile.setEnabled(cbFile.isSelected());

        btnAceptar.setText("Aceptar");

        btnCancelar.setText("Cancelar");

        GroupLayout layout = new GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(GroupLayout.LEADING)
            .add(layout.createSequentialGroup()
                .addContainerGap()
                .add(layout.createParallelGroup(GroupLayout.LEADING)
                    .add(GroupLayout.TRAILING, pnlCompresion, GroupLayout.DEFAULT_SIZE, GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .add(layout.createParallelGroup(GroupLayout.LEADING)
                        .add(GroupLayout.TRAILING, layout.createParallelGroup(GroupLayout.LEADING)
                            .add(layout.createSequentialGroup()
                                .add(layout.createParallelGroup(GroupLayout.LEADING)
                                    .add(cbFile)
                                    .add(cbVisible))
                                .add(63, 63, 63))
                            .add(GroupLayout.TRAILING, layout.createSequentialGroup()
                                .add(txtFile, GroupLayout.PREFERRED_SIZE, 170, GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(LayoutStyle.RELATED)
                                .add(btnFile, GroupLayout.PREFERRED_SIZE, 47, GroupLayout.PREFERRED_SIZE))
                            .add(layout.createSequentialGroup()
                                .add(layout.createParallelGroup(GroupLayout.LEADING)
                                    .add(GroupLayout.TRAILING, layout.createSequentialGroup()
                                        .add(lblIP)
                                        .add(150, 150, 150))
                                    .add(layout.createSequentialGroup()
                                        .add(txtIP, GroupLayout.PREFERRED_SIZE, 202, GroupLayout.PREFERRED_SIZE)
                                        .addPreferredGap(LayoutStyle.RELATED)))
                                .addPreferredGap(LayoutStyle.RELATED)
                                .add(layout.createParallelGroup(GroupLayout.LEADING)
                                    .add(lblPort)
                                    .add(txtPort, GroupLayout.PREFERRED_SIZE, 51, GroupLayout.PREFERRED_SIZE))))
                        .add(GroupLayout.TRAILING, layout.createSequentialGroup()
                            .add(btnAceptar)
                            .addPreferredGap(LayoutStyle.RELATED)
                            .add(btnCancelar)))
                    .add(pnlCaptura, GroupLayout.DEFAULT_SIZE, GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(GroupLayout.LEADING)
            .add(layout.createSequentialGroup()
                .addContainerGap()
                .add(pnlCaptura, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(LayoutStyle.RELATED)
                .add(pnlCompresion, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(LayoutStyle.RELATED)
                .add(layout.createParallelGroup(GroupLayout.TRAILING)
                    .add(layout.createSequentialGroup()
                        .add(lblIP)
                        .addPreferredGap(LayoutStyle.RELATED)
                        .add(txtIP, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
                    .add(layout.createSequentialGroup()
                        .add(lblPort)
                        .addPreferredGap(LayoutStyle.RELATED)
                        .add(txtPort, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE)))
                .add(14, 14, 14)
                .add(cbVisible)
                .addPreferredGap(LayoutStyle.RELATED)
                .add(cbFile)
                .addPreferredGap(LayoutStyle.RELATED)
                .add(layout.createParallelGroup(GroupLayout.BASELINE)
                    .add(txtFile, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE)
                    .add(btnFile))
                .addPreferredGap(LayoutStyle.RELATED)
                .add(layout.createParallelGroup(GroupLayout.BASELINE)
                    .add(btnCancelar)
                    .add(btnAceptar))
                .addContainerGap())
        );
        pack();


        Dimension d = Toolkit.getDefaultToolkit().getScreenSize();
        setLocation((d.width - getWidth()) / 2, (d.height - getHeight()) / 2);

        cbFile.addMouseListener(this);
        btnFile.addMouseListener(this);
        btnAceptar.addMouseListener(this);
        btnCancelar.addMouseListener(this);
    }

    /**
     * Evento <i>mouseClicked</i>. Comprueba qué elemento se pulsó y realiza la
     * acción correspondiente.
     * <p>Si se hace click sobre el checkBox correspondiente a la opción de
     * guardar en el fichero, se activan o desactiva el campo de texto y el botón
     * para abrir el diálogo de selección del fichero</p>
     * <p>Si se pulsa sobre el botón de selección de fichero, se abre el cuadro
     * de diálogo correspondiente</p>
     * <p>Si se pulsa sobre el botón <i>Cancelar</i>, se descartan los cambios</p>
     * <p>Si se pulsa sobre el botón <i>Aceptar</i>, se comprueba la integridad
     * de los datos y se guardan en el VideoSvrConf</p>
     * @param e MouseEvent
     */
    public void mouseClicked(MouseEvent e) {
        // Comprueba que se pulsó el botón izquierdo
        if (e.getButton() == MouseEvent.BUTTON1) {
            // Se hizo click sobre el checkBox del fichero
            if (e.getSource() == cbFile) {
                txtFile.setEnabled(cbFile.isSelected());
                btnFile.setEnabled(cbFile.isSelected());
                // Se pulsó sobre el botón de selección del fichero
            } else if (e.getSource() == btnFile) {
                // Se comprueba que está activo el botón
                if (btnFile.isEnabled()) {
                    int index = cmbCodec.getSelectedIndex();

                    // Se crea el diálogo de selección del fichero
                    FileSystemView fsv = FileSystemView.getFileSystemView();
                    JFileChooser fc = new JFileChooser(fsv.getHomeDirectory());

                    // Se crea un nuevo filtro basado en el tipo de multiplexado elegido
                    Filtro filtro = new Filtro(extensiones[index],
                                               descripciones[index]);
                    fc.addChoosableFileFilter(filtro);
                    fc.setFileFilter(filtro);

                    // Se muestra el diálogo y se pone el fichero elegido en el cuadro de texto
                    if (fc.showSaveDialog(this) == JFileChooser.APPROVE_OPTION) {
                        String fichero = fc.getSelectedFile().toString();
                        boolean conExt = false;
                        for (int i = 0; i < extensiones[index].length; i++) {
                            if (fichero.endsWith(extensiones[index][i])) {
                                conExt = true;
                                break;
                            }
                        }
                        if (! conExt)
                            fichero += extensiones[cmbCodec.getSelectedIndex()][0];
                        txtFile.setText(fichero);
                    }
                }
                // Se pulsó Cancelar
            } else if (e.getSource() == btnCancelar) {
                this.dispose();
                // Se pulsó Aceptar
            } else if (e.getSource() == btnAceptar) {
                // Se obtienen los datos
                int width = Integer.parseInt(txtWidth.getText());
                int height = Integer.parseInt(txtHeight.getText());
                int caching = ((Integer)spnCaching.getValue()).intValue();
                float fps = ((Double)spnFps.getValue()).floatValue();
                String serial = (String)cmbPuerto.getModel().getSelectedItem();
                String codec = (String)cmbCodec.getSelectedItem();
                int bitrate = 0;
                try {
                    bitrate = Integer.parseInt((String) cmbBitrate.getSelectedItem());
                } catch (Exception ex) {
                    Constantes.mensaje("El BitRate indicado es inválido ", ex);
                    return;
                }
                float scale = 0.0f;
                try {
                    scale = Float.parseFloat((String) cmbScale.getSelectedItem());
                } catch (Exception ex) {
                    Constantes.mensaje("La escala indicada es inválida ", ex);
                    return;
                }

                boolean display = cbVisible.isSelected();
                String muxName = mux[cmbMux.getSelectedIndex()];
                String file = ((cbFile.isSelected()) && (txtFile.getText().length() != 0))? txtFile.getText().replaceAll(" ", "&nbsp;") : null;
                String ip = txtIP.getText();
                int port = Integer.parseInt(txtPort.getText());

                // Comprobamos que los datos son correctos antes de escribirlos al objeto
                if ((width < 8) || (width > 1024) || (height < 8) || (height > 1024)) {
                    Constantes.mensaje("El tamaño " + width + "x" + height + " es inválido");
                    return;
                }

                if ((caching < 1) || (caching > 10000)) {
                    Constantes.mensaje("El tamaño de caché " + caching + " es inválido");
                    return;
                }

                if ((fps < 1) && (fps > 100)) {
                    Constantes.mensaje("El número de fps " + fps + " es inválido");
                    return;
                }

                // Comprueba IP y puerto
                String ip2 = ip.replaceAll("\\p{Punct}", "p");
                Pattern p = Pattern.compile("\\d{1,3}p\\d{1,3}p\\d{1,3}p\\d{1,3}");
                Matcher m = p.matcher(ip2);
                if (! m.matches()) {
                    Constantes.mensaje("La IP " + ip + " no es correcta");
                    return;
                }

                if ((port < 1025) || (port > 65535)) {
                    Constantes.mensaje("El número de puerto " + port + " es inválido");
                    return;
                }

                // Se establecen los valores obtenidos en el objeto VideoSvrConf
                vsc.setValues(width, height, caching, fps, codec, bitrate, scale, display, muxName, file, ip, port, serial);

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
