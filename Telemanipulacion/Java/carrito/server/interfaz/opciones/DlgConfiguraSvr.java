/**
 * Paquete que contiene las clases correspondientes a la interfaz de las opciones
 * del servidor
 */
package carrito.server.interfaz.opciones;

import java.awt.*;
import java.awt.event.*;
import java.util.*;

import javax.comm.*;
import javax.swing.*;

import carrito.configura.*;
import carrito.media.*;
import org.jdesktop.layout.*;


import org.jdesktop.layout.GroupLayout;
import org.jdesktop.layout.LayoutStyle;

/**
 * Clase que crea una interfaz para llevar a cabo la configuración del servidor
 * @author Néstor Morales Hernández
 * @version 1.0
 */
public class DlgConfiguraSvr extends JFrame implements MouseListener {
    // Variables de la interfaz
    private JButton btnAceptar;
    private JButton btnCancelar;
    private JButton btnConfigura;
    private JButton btnElimina;
    private JButton btnRestaura;
    private JComboBox cmbControlCamaras;
    private JComboBox cmbControlCarro;
    private JLabel lblControlCamaras;
    private JLabel lblControlCarro;
    private JList lstLista;
    private JPanel pnlCaptura;
    private JPanel pnlSerie;
    private JScrollPane scrScroll;
    private DefaultListModel model;

    /** Objeto que hace de interfaz entre todas las variables comunes a la aplicación */
    private Constantes cte;
    /** Array que contiene la lista de los puertos COM activos */
    private String[] puertos = null;
    /** Array que contiene la lista de dispositivos de captura de vídeo conectados */
    private VideoSvrConf[] dispositivos = null;

    /**
     * Constructor. Abre la configuración desde un fichero, obtiene la lista de
     * puertos COM activos, obtiene la lista de los dispositivos de captura y crea
     * la interfaz
     * @param cte Constantes
     */
    public DlgConfiguraSvr(Constantes cte) {
        this.cte = cte;
        // Obtiene la configuración del servidor desde el fichero de configuración
        // En caso de que este no exista, crea uno nuevo con los valores por defecto
        cte.openServer();

        // Obtiene la lista de puertos COM activos
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

        // Obtiene la lista de dispositivos de captura
        dispositivos = cte.getEmisores();

        // Crea la interfaz
        initComponents();
    }

    /**
     * Crea la interfaz de configuración del servidor
     */
    private void initComponents() {
        pnlCaptura = new JPanel();
        scrScroll = new JScrollPane();
        lstLista = new JList();
        btnConfigura = new JButton();
        btnElimina = new JButton();
        btnRestaura = new JButton();
        pnlSerie = new JPanel();
        lblControlCarro = new JLabel();
        lblControlCamaras = new JLabel();
        cmbControlCarro = new JComboBox();
        cmbControlCamaras = new JComboBox();
        btnCancelar = new JButton();
        btnAceptar = new JButton();
        model = new DefaultListModel();

        setIconImage(new ImageIcon("cars.jpg").getImage());
        setResizable(false);
        setTitle("Configuración del Servidor");
        setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
        pnlCaptura.setBorder(BorderFactory.createTitledBorder(
                "Propiedades de Captura"));
        lstLista.setModel(model);
        for (int i = 0; i < dispositivos.length; i++) {
            model.addElement(dispositivos[i]);
        }
        lstLista.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        lstLista.setSelectedIndex(0);
        scrScroll.setViewportView(lstLista);

        btnConfigura.setText("Configurar...");

        btnElimina.setText("Eliminar");

        btnRestaura.setText("Restaurar");

        GroupLayout pnlCapturaLayout = new GroupLayout(pnlCaptura);
        pnlCaptura.setLayout(pnlCapturaLayout);
        pnlCapturaLayout.setHorizontalGroup(
                pnlCapturaLayout.createParallelGroup(GroupLayout.LEADING)
                .add(pnlCapturaLayout.createSequentialGroup()
                     .addContainerGap()
                     .add(scrScroll, GroupLayout.PREFERRED_SIZE, 254,
                          GroupLayout.PREFERRED_SIZE)
                     .addPreferredGap(LayoutStyle.RELATED)
                     .add(pnlCapturaLayout.createParallelGroup(GroupLayout.
                LEADING, false)
                          .add(btnRestaura, GroupLayout.DEFAULT_SIZE,
                               GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                          .add(btnElimina, GroupLayout.DEFAULT_SIZE,
                               GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                          .add(btnConfigura, GroupLayout.DEFAULT_SIZE,
                               GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                     .addContainerGap(GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                );
        pnlCapturaLayout.setVerticalGroup(
                pnlCapturaLayout.createParallelGroup(GroupLayout.LEADING)
                .add(pnlCapturaLayout.createSequentialGroup()
                     .add(pnlCapturaLayout.createParallelGroup(GroupLayout.
                TRAILING, false)
                          .add(GroupLayout.LEADING, scrScroll, 0, 0,
                               Short.MAX_VALUE)
                          .add(GroupLayout.LEADING,
                               pnlCapturaLayout.createSequentialGroup()
                               .add(btnConfigura)
                               .addPreferredGap(LayoutStyle.RELATED)
                               .add(btnElimina)
                               .addPreferredGap(LayoutStyle.RELATED)
                               .add(btnRestaura)))
                     .addContainerGap(GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                );

        pnlSerie.setBorder(BorderFactory.createTitledBorder(
                "Configurar comunicaciones de serie"));
        lblControlCarro.setText("Control del Carro: ");

        lblControlCamaras.setText("Control de las C\u00e1maras: ");

        cmbControlCarro.setModel(new DefaultComboBoxModel(puertos));
        for (int i = 0; i < puertos.length; i++) {
            if (puertos[i].equals(cte.getCOMCarrito())) {
                cmbControlCarro.setSelectedIndex(i);
                break;
            }
        }

        cmbControlCamaras.setModel(new DefaultComboBoxModel(puertos));
        for (int i = 0; i < puertos.length; i++) {
            if (puertos[i].equals(cte.getCOMCamara())) {
                cmbControlCamaras.setSelectedIndex(i);
                break;
            }
        }

        GroupLayout pnlSerieLayout = new GroupLayout(pnlSerie);
        pnlSerie.setLayout(pnlSerieLayout);
        pnlSerieLayout.setHorizontalGroup(
            pnlSerieLayout.createParallelGroup(GroupLayout.LEADING)
            .add(pnlSerieLayout.createSequentialGroup()
                .addContainerGap()
                .add(pnlSerieLayout.createParallelGroup(GroupLayout.LEADING)
                    .add(lblControlCarro)
                    .add(lblControlCamaras))
                .add(83, 83, 83)
                .add(pnlSerieLayout.createParallelGroup(GroupLayout.TRAILING)
                    .add(cmbControlCarro, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE)
                    .add(cmbControlCamaras, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
                .addContainerGap(115, Short.MAX_VALUE))
        );
        pnlSerieLayout.setVerticalGroup(
            pnlSerieLayout.createParallelGroup(GroupLayout.LEADING)
            .add(pnlSerieLayout.createSequentialGroup()
                .add(pnlSerieLayout.createParallelGroup(GroupLayout.BASELINE)
                    .add(lblControlCarro)
                    .add(cmbControlCarro, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(LayoutStyle.RELATED)
                .add(pnlSerieLayout.createParallelGroup(GroupLayout.BASELINE)
                    .add(lblControlCamaras)
                    .add(cmbControlCamaras, GroupLayout.PREFERRED_SIZE, 22, GroupLayout.PREFERRED_SIZE))
                .addContainerGap(GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );

        btnCancelar.setText("Cancelar");

        btnAceptar.setText("Aceptar");

        GroupLayout layout = new GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(GroupLayout.LEADING)
            .add(layout.createSequentialGroup()
                .addContainerGap()
                .add(layout.createParallelGroup(GroupLayout.LEADING)
                    .add(layout.createParallelGroup(GroupLayout.LEADING)
                        .add(pnlSerie, GroupLayout.DEFAULT_SIZE, GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                        .add(layout.createSequentialGroup()
                            .add(pnlCaptura, GroupLayout.DEFAULT_SIZE, GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                            .addContainerGap()))
                    .add(GroupLayout.TRAILING, layout.createSequentialGroup()
                        .add(btnAceptar)
                        .addPreferredGap(LayoutStyle.RELATED)
                        .add(btnCancelar)
                        .add(32, 32, 32))))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(GroupLayout.LEADING)
            .add(layout.createSequentialGroup()
                .addContainerGap()
                .add(pnlCaptura, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(LayoutStyle.RELATED)
                .add(pnlSerie, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(LayoutStyle.RELATED)
                .add(layout.createParallelGroup(GroupLayout.BASELINE)
                    .add(btnAceptar)
                    .add(btnCancelar))
                .addContainerGap(GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );

        pack();
        Dimension d = Toolkit.getDefaultToolkit().getScreenSize();
        setLocation((d.width - getWidth()) / 2, (d.height - getHeight()) / 2);


        btnConfigura.addMouseListener(this);
        btnElimina.addMouseListener(this);
        btnRestaura.addMouseListener(this);
        btnAceptar.addMouseListener(this);
        btnCancelar.addMouseListener(this);
    }

    /**
     * Evento <i>mouseClicked</i>. Comprueba qué botón se pulsó y realiza la
     * acción correspondiente.
     * <p>Si se pulsa el botón <i>Configurar</i>, abre la ventana de configuración
     * para el dispositivo de captura seleccionado</p>
     * <p>Si se pulsa el botón <i>Eliminar</i>, se elimina el dispositivo de
     * captura seleccionado</p>
     * <p>Si se pulsa el botón <i>Restaurar</i>, se vuelve a obtener la lista de
     * dispositivos conectados</p>
     * <p>Si se pulsa el botón <i>Cancelar</i>, se mantiene la configuración
     * previa y se inicia la aplicación</p>
     * <p>Si se pulsa el botón <i>Aceptar</i>, se guarda la configuración en
     * el objeto Constantes y desde ahí, se guarda en el fichero de configuración.
     * Una vez hecho esto, se inicia la aplicación</p>
     * @param e MouseEvent
     */

    public void mouseClicked(MouseEvent e) {
        // Se pulsó el botón izquierdo
        if (e.getButton() == MouseEvent.BUTTON1) {
            // Se pulsó configurar
            if (e.getSource() == btnConfigura) {
                // Se comprueba que hay algún elemento seleccionado
                if (lstLista.getSelectedIndex() >= 0)
                    new DlgOpcionesCamaras(this, dispositivos[lstLista.getSelectedIndex()]).setVisible(true);
            // Se pulsó eliminar
            } else if (e.getSource() == btnElimina) {
                int seleccion = lstLista.getSelectedIndex();
                // Se comprueba que hay algún elemento seleccionado
                if (seleccion < 0)
                    return;
                // Elimina el elemento seleccionado
                model.remove(lstLista.getSelectedIndex());
                // Se modifica el foco de selección en la lista
                if (model.getSize() > 0) {
                    if (seleccion > 0)
                        seleccion--;
                    lstLista.setSelectedIndex(seleccion);
                }
            // Se pulsó restaurar
            } else if (e.getSource() == btnRestaura) {
                // Obtiene la lista de dispositivos multimedia conectados
                String dispName[] = Media.listaDispositivos();

                // Crea la lista de descriptores de dichos dispositivos
                dispositivos = new VideoSvrConf[dispName.length];
                for (int i = 0; i < dispositivos.length; i++) {
                    dispositivos[i] = new VideoSvrConf(dispName[i], i);
                }

                // Añade la lista al JList
                model.removeAllElements();
                for (int i = 0; i < dispositivos.length; i++) {
                    model.addElement(dispositivos[i]);
                }
                // Se pulsó el botón Cancelar
            } else if (e.getSource() == btnCancelar) {
                dispose();
                // Se pulsó aceptar
            } else if (e.getSource() == btnAceptar) {
                // Obtiene la lista final de dispositivos que se van a usar
                VideoSvrConf res[] = new VideoSvrConf[model.size()];
                for (int i = 0; i < res.length; i++) {
                    res[i] = (VideoSvrConf)model.elementAt(i);
                }
                // Se guarda la configuración en el objeto Constantes
                cte.setEmisores(res);
                cte.setCOMCarrito((String)cmbControlCarro.getSelectedItem());
                cte.setCOMCamara((String)cmbControlCamaras.getSelectedItem());
                // Se guarda en el fichero
                cte.saveServer();

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
