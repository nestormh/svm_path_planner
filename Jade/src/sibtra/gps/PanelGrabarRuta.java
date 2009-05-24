package sibtra.gps;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;

import javax.swing.Action;
import javax.swing.JButton;
import javax.swing.JComponent;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JPanel;

import sibtra.gps.GPSConnection;
import sibtra.gps.GpsEvent;
import sibtra.gps.GpsEventListener;

public class PanelGrabarRuta extends JPanel implements GpsEventListener,
        ActionListener {

    private GPSConnection gpsCon;

    private JButton jbGrabar;
    private JButton jbParar;
    private JFileChooser fc;
    private JLabel jlNpBT;
    private JLabel jlNpBE;
    private JLabel jlNpRT;
    private JLabel jlNpRE;
    private boolean cambioRuta = false;

    
    public PanelGrabarRuta(GPSConnection gpsc, Action actGrabar, Action actParar) {
    	super();
    	if(gpsc==null)
    		throw new IllegalArgumentException("conexión a GPS no puede ser null");
    	JComponent ja;

    	jbGrabar = new JButton("Grabar");
    	jbGrabar.setAction(actGrabar);
    	jbGrabar.addActionListener(this);
    	add(jbGrabar);

    	jbParar = new JButton("Parar");
    	jbParar.setAction(actParar);
    	jbParar.setEnabled(false);
    	jbParar.addActionListener(this);
    	add(jbParar);

    	ja = jlNpBT = new JLabel("BT: ?????");
    	ja.setEnabled(true);
    	add(ja);
    	ja = jlNpBE = new JLabel("BE: ?????");
    	ja.setEnabled(true);
    	add(ja);
    	ja = jlNpRT = new JLabel("RT: ?????");
    	ja.setEnabled(false);
    	add(ja);
    	ja = jlNpRE = new JLabel("RE: ?????");
    	ja.setEnabled(false);
    	add(ja);

        gpsCon.addGpsEventListener(this);

        //elegir fichero
        fc = new JFileChooser(new File("./Rutas"));

    }

    public PanelGrabarRuta(GPSConnection gpsc) {
    	this(gpsc,null,null);
    }
    
    public void setAccionGrabar(Action act) {
    	jbGrabar.setAction(act);
    }

    public void setAccionParar(Action act) {
    	jbParar.setAction(act);
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
            cambioRuta = false;
        }
    }

    public void actionPerformed(ActionEvent ae) {
        if (ae.getSource() == jbGrabar) {
            //comienza la grabación
            cambioRuta = true;
            gpsCon.startRuta();
            //el startRuta crea una nueva ruta, nos actualizamos
            jbGrabar.setEnabled(false);
            jbParar.setEnabled(true);
        }
        if (ae.getSource() == jbParar) {
            gpsCon.stopRuta();
            int devuelto = fc.showSaveDialog(this);
            if (devuelto == JFileChooser.APPROVE_OPTION) {
                File file = fc.getSelectedFile();
                gpsCon.saveRuta(file.getAbsolutePath());
            }
            jbGrabar.setEnabled(true);
        }
    }

}
