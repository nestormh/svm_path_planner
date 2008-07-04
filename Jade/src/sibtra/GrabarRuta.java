package sibtra;

import java.awt.BorderLayout;
import java.awt.Dimension;

import javax.swing.JFrame;

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

public class GrabarRuta implements GpsEventListener, IMUEventListener {
	
	private GPSConnection gpsCon;
	private JFrame ventGData;
	private PanelMuestraGPSData PMGPS;
	private JFrame ventIMU;
	private PanelMuestraAngulosIMU pmai;
	private ConexionSerialIMU csi;
	private JFrame ventRuta;
	private PanelMuestraRuta pmr;
	private AngulosIMU ultimoAngulo;

	public GrabarRuta(String[] args) {
		if(args.length<2) {
			System.err.println("Necesarios dos parámetros con los puertos de GPS e IMU");
			System.exit(1);
		}
		
		//comunicación con GPS
		gpsCon=new SimulaGps(args[0]).getGps();
		if(gpsCon==null) {
			System.err.println("No se obtuvo GPSConnection");
			System.exit(1);
		}

		//conexión IMU
		csi=new ConexionSerialIMU();
		if(!csi.ConectaPuerto(args[1])) {
			System.err.println("Problema en conexión serial con la IMU");
			System.exit(1);
		}

		//VEntana datos gps
		ventGData=new JFrame("Datos GPS");
		PMGPS=new PanelMuestraGPSData();
		PMGPS.actualizaPunto(new GPSData()); 
		gpsCon.addGpsEventListener(PMGPS);

		ventGData.getContentPane().add(PMGPS,BorderLayout.CENTER);
		ventGData.pack();
		ventGData.setVisible(true);

		//Creamos ventana para IMU
		ventIMU=new JFrame("Datos IMU");
		pmai=new PanelMuestraAngulosIMU();
		pmai.actualizaAngulo(new AngulosIMU(0,0,0,0));
		ventIMU.add(pmai);
		csi.addIMUEventListener(pmai);

		ventIMU.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		ventIMU.pack();
		ventIMU.setVisible(true);

		//Creamos ventana para la ruta
		ventRuta=new JFrame("Ruta");
		pmr=new PanelMuestraRuta(gpsCon.getRutaEspacial());
		gpsCon.addGpsEventListener(pmr);
		ventRuta.getContentPane().add(pmr,BorderLayout.CENTER);
//		ventRuta.pack();
		ventRuta.setSize(new Dimension(800,600));
		ventRuta.setVisible(true);
		
	}


	public void handleGpsEvent(GpsEvent ev) {
		//ponemos el último ángulo de la IMU en el último dato del GPS
		ev.getNuevoPunto().setAgulosIMU(ultimoAngulo);
		if(pmr.getRuta()==null && gpsCon.getRutaTemporal()!=null)
			pmr.setRuta(gpsCon.getRutaTemporal());
	}

	public void handleIMUEvent(IMUEvent ev) {
		ultimoAngulo=new AngulosIMU(ev.getAngulos());
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		GrabarRuta gr=new GrabarRuta(args);
		
	}

}
