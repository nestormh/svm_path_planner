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
import javax.swing.JWindow;

import sibtra.gps.GPSConnection;
import sibtra.gps.GPSData;
import sibtra.gps.GpsEvent;
import sibtra.gps.GpsEventListener;
import sibtra.gps.PanelMuestraGPSData;
import sibtra.gps.PanelMuestraRuta;
import sibtra.gps.SimulaGps;
import sibtra.imu.AngulosIMU;
import sibtra.imu.ConexionSerialIMU;
import sibtra.imu.IMUEvent;
import sibtra.imu.IMUEventListener;
import sibtra.imu.PanelMuestraAngulosIMU;

public class GrabarRuta implements GpsEventListener, 
ActionListener {
	
	private GPSConnection gpsCon;
	private JFrame ventGData;
	private PanelMuestraGPSData PMGPS;
	private JFrame ventIMU;
	private PanelMuestraAngulosIMU pmai;
	private ConexionSerialIMU csi;
	private JFrame ventRuta;
	private PanelMuestraRuta pmr;
	private JButton jbGrabar;
	private JButton jbParar;
	private JFileChooser fc;
	private JLabel jlNpBT;
	private JLabel jlNpBE;
	private JLabel jlNpRT;
	private JLabel jlNpRE;
	private boolean cambioRuta=false;

	public GrabarRuta(String[] args) {
		if(args.length<2) {
			System.err.println("Necesarios dos parámetros con los puertos de GPS e IMU");
			System.exit(1);
		}
		
		System.out.println("Abrimos conexión IMU");
		csi=new ConexionSerialIMU();
		if(!csi.ConectaPuerto(args[1],5)) {
			System.err.println("Problema en conexión serial con la IMU");
			System.exit(1);
		}
		
		//comunicación con GPS
		gpsCon=new SimulaGps(args[0]).getGps();
		if(gpsCon==null) {
			System.err.println("No se obtuvo GPSConnection");
			System.exit(1);
		}
		gpsCon.setCsIMU(csi);
		
		//VEntana datos gps
		ventGData=new JFrame("Datos GPS");
		PMGPS=new PanelMuestraGPSData(true);
		PMGPS.actualizaPunto(new GPSData()); 

		ventGData.getContentPane().add(PMGPS,BorderLayout.CENTER);
		ventGData.pack();
		ventGData.setVisible(true);

		//Creamos ventana para la ruta
		ventRuta=new JFrame("Ruta");
		pmr=new PanelMuestraRuta(gpsCon.getBufferRutaEspacial());
		ventRuta.getContentPane().add(pmr,BorderLayout.CENTER);
		{ //sur ventana
			JComponent ja;
			
			JPanel jpSur=new JPanel();

			jbGrabar=new JButton("Grabar");
			jbGrabar.addActionListener(this);
			jpSur.add(jbGrabar);

			jbParar=new JButton("Parar");
			jbParar.setEnabled(false);
			jbParar.addActionListener(this);
			jpSur.add(jbParar);
			
			ja=jlNpBT=new JLabel("BT: ?????"); ja.setEnabled(true); jpSur.add(ja); 
			ja=jlNpBE=new JLabel("BE: ?????"); ja.setEnabled(true); jpSur.add(ja); 
			ja=jlNpRT=new JLabel("RT: ?????"); ja.setEnabled(false); jpSur.add(ja); 
			ja=jlNpRE=new JLabel("RE: ?????"); ja.setEnabled(false); jpSur.add(ja); 
			
			ventRuta.getContentPane().add(jpSur,BorderLayout.SOUTH);
		}
//		ventRuta.pack();
		ventRuta.setSize(new Dimension(800,600));
		ventRuta.setVisible(true);
		//conecto manejador cuando todas las ventanas están creadas
		gpsCon.addGpsEventListener(this);
		gpsCon.addGpsEventListener(PMGPS);
		gpsCon.addGpsEventListener(pmr);

//		try { Thread.sleep(10000); } catch (Exception e) {	}
		
		//Creamos ventana para IMU
		ventIMU=new JFrame("Datos IMU");
		pmai=new PanelMuestraAngulosIMU();
		pmai.actualizaAngulo(new AngulosIMU(0,0,0,0));
		ventIMU.add(pmai);

		ventIMU.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		ventIMU.pack();
		ventIMU.setVisible(true);
		//conecto manejador cuando todas las ventanas están creadas
		csi.addIMUEventListener(pmai);


		//elegir fichero
		fc=new JFileChooser(new File("./Rutas"));

	}


	public void handleGpsEvent(GpsEvent ev) {
		//actualizamos el número de puntos
		jlNpBE.setText(String.format("BE: %5d", gpsCon.getBufferEspacial().getNumPuntos()));
		jlNpBT.setText(String.format("BT: %5d", gpsCon.getBufferTemporal().getNumPuntos()));
		if(gpsCon.getBufferRutaEspacial()!=null) {
			jlNpRE.setText(String.format("RE: %5d", gpsCon.getBufferRutaEspacial().getNumPuntos()));
			jlNpRE.setEnabled(true);
		}
		if(gpsCon.getBufferRutaTemporal()!=null) {
			jlNpRT.setText(String.format("RT: %5d", gpsCon.getBufferRutaTemporal().getNumPuntos()));
			jlNpRT.setEnabled(true);
		}
		if(cambioRuta) {
			System.out.println("Conectamos la ruta espacial");
			pmr.setRuta(gpsCon.getBufferRutaEspacial());
			cambioRuta=false;
		}
	}


	public void actionPerformed(ActionEvent ae) {
		if(ae.getSource()==jbGrabar) {
			//comienza la grabación
			cambioRuta=true;
			gpsCon.startRuta();
			jbGrabar.setEnabled(false);
			jbParar.setEnabled(true);
		}
		if(ae.getSource()==jbParar) {
			gpsCon.stopRuta();
			int devuelto=fc.showSaveDialog(ventRuta);
			if(devuelto==JFileChooser.APPROVE_OPTION) {
				File file=fc.getSelectedFile();
				gpsCon.saveRuta(file.getAbsolutePath());
			}
			jbGrabar.setEnabled(true);
		}
	}


	/**
	 * @param args
	 */
	public static void main(String[] args) {
		GrabarRuta gr=new GrabarRuta(args);
		
	}


}
