/**
 * 
 */
package sibtra.gps;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import javax.swing.border.Border;

import Jama.Matrix;

/**
 * {@link JPanel} para mostrar los datos más significativos de un GPSData.
 * @author alberto
 *
 */
public class PanelMuestraGPSData extends JPanel implements GpsEventListener {

	private JLabel jlHora;
	private JLabel jlLatitud;
	private JLabel jlLongitud;
	private JLabel jlRms;
	private JLabel jlNumSat;
	private JLabel jlEdad;
	private JLabel jlCoordLocales;
	private JLabel jlCoordECEF;
	public PanelMuestraGPSData() {
		setLayout(new GridLayout(0,3)); //empezamos con 3 columnas
//		altura=aCopiar.altura;
//		angulo=aCopiar.angulo;
//		cadenaNMEA=aCopiar.cadenaNMEA;
//		coordECEF=(Matrix)aCopiar.coordECEF.clone();
//		coordLocal=(Matrix)aCopiar.coordLocal.clone();
//		desvAltura=aCopiar.desvAltura;
//		desvEjeMayor=aCopiar.desvEjeMayor;
//		desvEjeMenor=aCopiar.desvEjeMenor;
//		desvLatitud=aCopiar.desvLatitud;
//		desvLongitud=aCopiar.desvLongitud;
//		hdgPoloM=aCopiar.hdgPoloM;
//		hdgPoloN=aCopiar.hdgPoloN;
//		hdoP=aCopiar.hdoP;
//		hGeoide=aCopiar.hGeoide;
//		msL=aCopiar.msL;
//		orientacionMayor=aCopiar.orientacionMayor;
//		pdoP=aCopiar.pdoP;
//		sysTime=aCopiar.sysTime;
//		vdoP=aCopiar.vdoP;
//		velocidad=aCopiar.velocidad;
//		velocidadGPS=aCopiar.velocidadGPS;		
		
		Border blackline = BorderFactory.createLineBorder(Color.black);
		Font Grande;
		JLabel jla; //variable para poner JLable actual
		{ //hora
			jlHora=jla=new JLabel("??:??:??.??");
		    Grande = jla.getFont().deriveFont(20.0f);
			jla.setBorder(BorderFactory.createTitledBorder(
				       blackline, "Hora"));
		    jla.setFont(Grande);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
			add(jla);
		}

		{ //Latitud
			jlLatitud=jla=new JLabel("+?? ??.?????");
			jla.setBorder(BorderFactory.createTitledBorder(
					       blackline, "Latitud"));
		    jla.setFont(Grande);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
			add(jla);
		}

		{ //Longitud
			jlLongitud=jla=new JLabel("+??? ??.?????");
			jla.setBorder(BorderFactory.createTitledBorder(
					       blackline, "Longitud"));
		    jla.setFont(Grande);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
			add(jla);
		}

		{ //RMS
			jlRms=jla=new JLabel("?.?");
			jla.setBorder(BorderFactory.createTitledBorder(
					       blackline, "RMS"));
		    jla.setFont(Grande);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
			add(jla);
		}

		{ //Número satélites
			jlNumSat=jla=new JLabel("?");
			jla.setBorder(BorderFactory.createTitledBorder(
					       blackline, "Num Satelites"));
		    jla.setFont(Grande);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
			add(jla);
		}

		{ //Edad correción diferencial
			jlEdad=jla=new JLabel("??? sg");
			jla.setBorder(BorderFactory.createTitledBorder(
					       blackline, "Edad Correccion"));
		    jla.setFont(Grande);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
			add(jla);
		}

		{ //Coordenadas locales
			jlCoordLocales=jla=new JLabel("(???.??, ???.??, ???.??)");
			jla.setBorder(BorderFactory.createTitledBorder(
					       blackline, "Locales"));
		    jla.setFont(Grande);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
			add(jla);
		}
		
		{ //Coordenadas ECEF
			jlCoordECEF=jla=new JLabel("(???.??, ???.??, ???.??)");
			jla.setBorder(BorderFactory.createTitledBorder(
					       blackline, "ECEF"));
		    jla.setFont(Grande);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
			add(jla);
		}
		
	
	}
	
	public void actualizaPunto(GPSData pto) {
		if(pto==null) {
			//no están las 3 distancias, regresamos
			jlHora.setEnabled(false);
			jlLatitud.setEnabled(false);
			jlLongitud.setEnabled(false);
			jlRms.setEnabled(false);
			jlNumSat.setEnabled(false);
			jlEdad.setEnabled(false);
			jlCoordLocales.setEnabled(false);
			jlCoordECEF.setEnabled(false);
		} else {
			jlHora.setEnabled(true);
			jlLatitud.setEnabled(true);
			jlLongitud.setEnabled(true);
			jlRms.setEnabled(true);
			jlNumSat.setEnabled(true);
			jlEdad.setEnabled(true);
			jlCoordLocales.setEnabled(true);
			jlCoordECEF.setEnabled(true);
			jlHora.setText(pto.getHora());
			jlLatitud.setText(pto.getLatitudText());
			jlLongitud.setText(pto.getLongitudText());
			jlRms.setText(String.format("%2.1f", pto.getRms()));
			jlNumSat.setText(String.format("%1d", pto.getSatelites()));
			jlEdad.setText(String.format("%3.0f", pto.getAge()));
			Matrix cl=pto.getCoordLocal();
			if(cl!=null && cl.get(0, 0)<Double.MAX_VALUE) {
				jlCoordLocales.setText(String.format("(%3.3f %3.3f %3.3f)"
						, cl.get(0,0), cl.get(1,0), cl.get(2,0)));
			} else jlCoordLocales.setEnabled(false);
			Matrix ce=pto.getCoordECEF();
			if(ce!=null && ce.get(0, 0)<Double.MAX_VALUE) {
				jlCoordECEF.setText(String.format("(%3.3f  %3.3f %3.3f)"
						, ce.get(0,0), ce.get(1,0), ce.get(2,0)));
			} else jlCoordECEF.setEnabled(false);
		}
		//programamos la actualizacion de la ventana
		SwingUtilities.invokeLater(new Runnable() {
			public void run() {
				repaint();
			}
		});


	}

	/* (non-Javadoc)
	 * @see sibtra.gps.GpsEventListener#handleGpsEvent(sibtra.gps.GpsEvent)
	 */
	public void handleGpsEvent(GpsEvent ev) {
		if(ev!=null)
			actualizaPunto(ev.getNuevoPunto());
	}

	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		JFrame ventanaPrincipal=new JFrame("PanelMuestraGPSData");
		PanelMuestraGPSData PMGPS=new PanelMuestraGPSData();
		PMGPS.actualizaPunto(new GPSData()); 

		final GPSConnection gpsc=new SimulaGps("/dev/ttyUSB0").getGps();
		if(gpsc!=null)
			gpsc.addGpsEventListener(PMGPS);
		else {
			System.out.println("No se obtuvo GPSConnection");
			System.exit(1);
		}
		gpsc.startRuta();
		
		ventanaPrincipal.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		ventanaPrincipal.getContentPane().add(PMGPS,BorderLayout.CENTER);
		JButton jbSalvar=new JButton("Salvar");
		jbSalvar.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				gpsc.stopRutaAndSave("Prueba1.gps");
			}
		}
		);
		ventanaPrincipal.getContentPane().add(jbSalvar,BorderLayout.PAGE_END);
		
		ventanaPrincipal.setSize(new Dimension(800,400));
		ventanaPrincipal.setVisible(true);
		
		
	}

}
