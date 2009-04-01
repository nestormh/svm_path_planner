package sibtra;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;

import javax.swing.JButton;
import javax.swing.JComponent;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;

import sibtra.controlcarro.ControlCarro;
import sibtra.controlcarro.VentanaCoche;
import sibtra.gps.GPSConnectionTriumph;
import sibtra.gps.GpsEvent;
import sibtra.gps.GpsEventListener;
import sibtra.gps.PanelMuestraRuta;
import sibtra.gps.VentanaGPSTriumph;
import sibtra.imu.ConexionSerialIMU;
import sibtra.imu.VentanaIMU;
import sibtra.util.EligeSerial;

public class GrabarRuta implements GpsEventListener,
        ActionListener {

    private GPSConnectionTriumph gpsCon;
    private ConexionSerialIMU csi;
    private ControlCarro contCarro;

    private VentanaGPSTriumph ventGData;
    private VentanaIMU ventIMU;
    private VentanaCoche ventCoche;

//    private PanelMuestraGPSData PMGPS;
//    private PanelMuestraAngulosIMU pmai;
    private JFrame ventRuta;
    private PanelMuestraRuta pmr;
    private JButton jbGrabar;
    private JButton jbParar;
    private JFileChooser fc;
    private JLabel jlNpBT;
    private JLabel jlNpBE;
    private JLabel jlNpRT;
    private JLabel jlNpRE;
    private boolean cambioRuta = false;
//    private JPanel jpCentral;

    /** @param args primer argumento puerto del GPS, segundo puerto del la IMU */
    public GrabarRuta(String[] args) {
        if (args.length < 2) {
            System.err.println("Necesarios dos parámetros con los puertos de GPS e IMU");
            System.exit(1);
        }

        System.out.println("Abrimos conexión IMU");
        csi = new ConexionSerialIMU();
        if (!csi.ConectaPuerto(args[1], 5)) {
            System.err.println("Problema en conexión serial con la IMU");
            System.exit(1);
        }

        //comunicación con GPS
        System.out.println("Abrimos conexión GPS");
        try {
            gpsCon = new GPSConnectionTriumph(args[0]);
        } catch (Exception e) {
            System.err.println("Promblema a crear GPSConnection:" + e.getMessage());
            System.exit(1);
        }
        if (gpsCon == null) {
            System.err.println("No se obtuvo GPSConnection");
            System.exit(1);
        }
        gpsCon.setCsIMU(csi);

        //Conectamos Carro
        System.out.println("Abrimos conexión al Carro");
        contCarro = new ControlCarro(args[2]);        
        gpsCon.setCsCARRO(contCarro);

        //Abrimos las ventanas
        //VEntana datos gps
        ventGData=new VentanaGPSTriumph(gpsCon);
        ventGData.pack();
        ventGData.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        ventGData.setVisible(true);

        //Creamos ventana para IMU
        ventIMU=new VentanaIMU(csi);
        ventIMU.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        ventIMU.pack();
        ventIMU.setVisible(true);

        //ventana del Coche
        ventCoche = new VentanaCoche(contCarro);
        ventCoche.pack();
        ventCoche.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        ventCoche.setVisible(true);
        
        //Creamos ventana para la ruta
        ventRuta = new JFrame("Graba Ruta");
        pmr = new PanelMuestraRuta(gpsCon.getBufferRutaEspacial());
        ventRuta.getContentPane().add(pmr, BorderLayout.CENTER);
        { //sur ventana
            JComponent ja;

            JPanel jpSur = new JPanel();

            jbGrabar = new JButton("Grabar");
            jbGrabar.addActionListener(this);
            jpSur.add(jbGrabar);

            jbParar = new JButton("Parar");
            jbParar.setEnabled(false);
            jbParar.addActionListener(this);
            jpSur.add(jbParar);

            ja = jlNpBT = new JLabel("BT: ?????");
            ja.setEnabled(true);
            jpSur.add(ja);
            ja = jlNpBE = new JLabel("BE: ?????");
            ja.setEnabled(true);
            jpSur.add(ja);
            ja = jlNpRT = new JLabel("RT: ?????");
            ja.setEnabled(false);
            jpSur.add(ja);
            ja = jlNpRE = new JLabel("RE: ?????");
            ja.setEnabled(false);
            jpSur.add(ja);

            ventRuta.getContentPane().add(jpSur, BorderLayout.SOUTH);
        }
//		ventRuta.pack();
        ventRuta.setSize(new Dimension(800, 600));
        ventRuta.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        ventRuta.setVisible(true);
        //conecto manejador cuando todas las ventanas están creadas
        gpsCon.addGpsEventListener(this);
        gpsCon.addGpsEventListener(pmr);

//		try { Thread.sleep(10000); } catch (Exception e) {	}

        //elegir fichero
        fc = new JFileChooser(new File("./Rutas"));

    }

    public void handleGpsEvent(GpsEvent ev) {
        //actualizamos el número de puntos
        jlNpBE.setText(String.format("BE: %5d", gpsCon.getBufferEspacial().getNumPuntos()));
        jlNpBT.setText(String.format("BT: %5d", gpsCon.getBufferTemporal().getNumPuntos()));
        if (gpsCon.getBufferRutaEspacial() != null) {
            jlNpRE.setText(String.format("RE: %5d", gpsCon.getBufferRutaEspacial().getNumPuntos()));
            jlNpRE.setEnabled(true);
        }
        if (gpsCon.getBufferRutaTemporal() != null) {
            jlNpRT.setText(String.format("RT: %5d", gpsCon.getBufferRutaTemporal().getNumPuntos()));
            jlNpRT.setEnabled(true);
        }
        if (cambioRuta) {
            System.out.println("Conectamos la ruta espacial");
            pmr.setRuta(gpsCon.getBufferRutaEspacial());
            cambioRuta = false;
        }
    }

    public void actionPerformed(ActionEvent ae) {
        if (ae.getSource() == jbGrabar) {
            //comienza la grabación
            cambioRuta = true;
            gpsCon.startRuta();
            //el startRuta crea una nueva ruta, nos actualizamos
            pmr.setRuta(gpsCon.getBufferRutaEspacial());
            jbGrabar.setEnabled(false);
            jbParar.setEnabled(true);
        }
        if (ae.getSource() == jbParar) {
            gpsCon.stopRuta();
            int devuelto = fc.showSaveDialog(ventRuta);
            if (devuelto == JFileChooser.APPROVE_OPTION) {
                File file = fc.getSelectedFile();
                gpsCon.saveRuta(file.getAbsolutePath());
            }
            jbGrabar.setEnabled(true);
        }
    }

    /**
     * @param args
     */
    public static void main(String[] args) {
        String[] puertos;
        if (args == null || args.length < 3) {
            //no se han pasado argumentos, pedimos los puertos interactivamente
            String[] titulos = {"GPS", "IMU", "Coche"};
            puertos = new EligeSerial(titulos).getPuertos();
            if (puertos == null) {
                System.err.println("No se asignaron los puertos seriales");
                System.exit(1);
            }
        } else {
            puertos = args;
        }
        GrabarRuta gr = new GrabarRuta(puertos);

    }
}
