/**
 * 
 */
package sibtra.gps;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Font;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.text.SimpleDateFormat;
import java.util.Date;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;
import javax.swing.border.Border;

import sibtra.util.EligeSerial;
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
	private JLabel jlAltura;
	private JLabel jlHGeoide;
	private JLabel jlAlturaGPS;
	private JLabel jlCoordECEFx;
	private JLabel jlCoordECEFy;
	private JLabel jlCoordECEFz;
	private JLabel jlYaw;
	private JLabel jlAngulo;
	private JLabel jlDMag;
	private JCheckBox jcbSoloEspa;
	
	/**
	 * constructor por defecto. Se actualiza con todos los puntos.
	 */
	public PanelMuestraGPSData() {
		this(false);
	}
	
	/**
	 * Constructor donde indicamos si sólo queremos actualizarnos con los espaciales
	 * @param soloEspaciales
	 */
	public PanelMuestraGPSData(boolean soloEspaciales) {
		JPanel jpCentro=new JPanel(new GridLayout(0,3)); //empezamos con 3 columnas
		setLayout(new BorderLayout());
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
			jpCentro.add(jla);
		}

		{ //Latitud
			jlLatitud=jla=new JLabel("+?? ??.?????");
			jla.setBorder(BorderFactory.createTitledBorder(
					       blackline, "Latitud"));
		    jla.setFont(Grande);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
			jpCentro.add(jla);
		}

		{ //Longitud
			jlLongitud=jla=new JLabel("+??? ??.?????");
			jla.setBorder(BorderFactory.createTitledBorder(
					       blackline, "Longitud"));
		    jla.setFont(Grande);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
			jpCentro.add(jla);
		}

		{ //RMS
			jlRms=jla=new JLabel("?.?");
			jla.setBorder(BorderFactory.createTitledBorder(
					       blackline, "RMS"));
		    jla.setFont(Grande);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
			jpCentro.add(jla);
		}

		{ //Número satélites
			jlNumSat=jla=new JLabel("?");
			jla.setBorder(BorderFactory.createTitledBorder(
					       blackline, "Num Satelites"));
		    jla.setFont(Grande);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
			jpCentro.add(jla);
		}

		{ //Edad correción diferencial
			jlEdad=jla=new JLabel("??? sg");
			jla.setBorder(BorderFactory.createTitledBorder(
					       blackline, "Edad Correccion"));
		    jla.setFont(Grande);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
			jpCentro.add(jla);
		}

		{ //Coordenadas locales
			jlCoordLocales=jla=new JLabel("(???.??, ???.??, ???.??)");
			jla.setBorder(BorderFactory.createTitledBorder(
					       blackline, "Locales"));
		    jla.setFont(Grande);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
			jpCentro.add(jla);
		}
		
		{ //Coordenadas ECEF X
			jlCoordECEFx=jla=new JLabel("???????.??");
			jla.setBorder(BorderFactory.createTitledBorder(
					       blackline, "ECEF X"));
		    jla.setFont(Grande);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
			jpCentro.add(jla);
		}

		{ //Coordenadas ECEF Y
			jlCoordECEFy=jla=new JLabel("???????.??");
			jla.setBorder(BorderFactory.createTitledBorder(
					       blackline, "ECEF Y"));
		    jla.setFont(Grande);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
			jpCentro.add(jla);
		}

		{ //Coordenadas ECEF Z
			jlCoordECEFz=jla=new JLabel("???????.??");
			jla.setBorder(BorderFactory.createTitledBorder(
					       blackline, "ECEF X"));
		    jla.setFont(Grande);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
			jpCentro.add(jla);
		}

		{//altura
			jlAltura=jla=new JLabel("+????.??");
			jla.setBorder(BorderFactory.createTitledBorder(
					       blackline, "Altura"));
		    jla.setFont(Grande);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
			jpCentro.add(jla);
		}
	
		{//altura sobre geoide
			jlHGeoide=jla=new JLabel("+????.??");
			jla.setBorder(BorderFactory.createTitledBorder(
					       blackline, "Alt. sobre Geoide"));
		    jla.setFont(Grande);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
//			jpCentro.add(jla);
		}
		{//altura
			jlAlturaGPS=jla=new JLabel("+????.??");
			jla.setBorder(BorderFactory.createTitledBorder(
					       blackline, "Altura GPS"));
		    jla.setFont(Grande);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
			jpCentro.add(jla);
		}
		{//Yaw
			jlYaw=jla=new JLabel("+????.??");
			jla.setBorder(BorderFactory.createTitledBorder(
					       blackline, "Yaw IMU"));
		    jla.setFont(Grande);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
			jpCentro.add(jla);
		}
		{//Angulo calculado
			jlAngulo=jla=new JLabel("+????.??");
			jla.setBorder(BorderFactory.createTitledBorder(
					       blackline, "Angulo Calc."));
		    jla.setFont(Grande);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
			jpCentro.add(jla);
		}
		{//Diff angulos
			jlDMag=jla=new JLabel("+????.??");
			jla.setBorder(BorderFactory.createTitledBorder(
					       blackline, "D. Magnetica"));
		    jla.setFont(Grande);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
			jpCentro.add(jla);
		}
		add(jpCentro,BorderLayout.CENTER);
		jcbSoloEspa=new JCheckBox("Sólo datos espaciales");
		jcbSoloEspa.setSelected(soloEspaciales);
		add(jcbSoloEspa,BorderLayout.SOUTH);
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
			jlCoordECEFx.setEnabled(false);
			jlCoordECEFy.setEnabled(false);
			jlCoordECEFz.setEnabled(false);
			jlAltura.setEnabled(false);
			jlHGeoide.setEnabled(false);
			jlAlturaGPS.setEnabled(false);
			jlYaw.setEnabled(false);
			jlAngulo.setEnabled(false);
			jlDMag.setEnabled(false);
		} else {
			jlHora.setEnabled(true);
			jlLatitud.setEnabled(true);
			jlLongitud.setEnabled(true);
			jlRms.setEnabled(true);
			jlNumSat.setEnabled(true);
			jlEdad.setEnabled(true);
			jlCoordLocales.setEnabled(true);
			jlCoordECEFx.setEnabled(true);
			jlCoordECEFy.setEnabled(true);
			jlCoordECEFz.setEnabled(true);
			jlAltura.setEnabled(true);
			jlHGeoide.setEnabled(true);
			jlAlturaGPS.setEnabled(true);
			jlYaw.setEnabled(true);
			//Nuevos valores
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
			} else 
				jlCoordLocales.setEnabled(false);
			Matrix ce=pto.getCoordECEF();
			if(ce!=null && ce.get(0, 0)<Double.MAX_VALUE) {
				jlCoordECEFx.setText(String.format("%10.3f",ce.get(0,0)));
				jlCoordECEFy.setText(String.format("%10.3f",ce.get(1,0)));
				jlCoordECEFz.setText(String.format("%10.3f",ce.get(2,0)));
			} else {
				jlCoordECEFx.setEnabled(false);
				jlCoordECEFy.setEnabled(false);
				jlCoordECEFz.setEnabled(false);
			}
			jlAltura.setText(String.format("%+8.2f", pto.getHGeoide()+pto.getMSL()));
			jlHGeoide.setText(String.format("%+8.2f", pto.getHGeoide()));
			jlAlturaGPS.setText(String.format("%+8.2f", pto.getMSL()));
			if(pto.getAngulosIMU()!=null) {
				jlYaw.setText(String.format("%+8.2f", pto.getAngulosIMU().getYaw()));
				jlYaw.setEnabled(true);
				jlDMag.setText(String.format("%+8.2f", 
						Math.toDegrees(pto.getAngulo())-pto.getAngulosIMU().getYaw()));
				jlDMag.setEnabled(true);
			} else {
				jlYaw.setEnabled(false);
				jlDMag.setEnabled(false);
			}
			jlAngulo.setText(String.format("%+8.2f", Math.toDegrees(pto.getAngulo())));
			jlAngulo.setEnabled(true);
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
		//atendemos sólo un tipo de evento según el tipo
		if(ev!=null) {
			if(jcbSoloEspa.isSelected() && ev.isEspacial())
				actualizaPunto(ev.getNuevoPunto());
			if(!jcbSoloEspa.isSelected() && !ev.isEspacial())
				actualizaPunto(ev.getNuevoPunto());				
		}
	}

	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		String[] puertos;
		if(args==null || args.length<1) {
			//no se han pasado argumentos, pedimos los puertos interactivamente
			String[] titulos={"GPS"};			
			puertos=new EligeSerial(titulos).getPuertos();
			if(puertos==null) {
				System.err.println("No se asignaron los puertos seriales");
				System.exit(1);
			}
		} else puertos=args;
		

		JFrame ventanaPrincipal=new JFrame("PanelMuestraGPSData");
		PanelMuestraGPSData PMGPS=new PanelMuestraGPSData();
		PMGPS.actualizaPunto(new GPSData()); 

		final GPSConnection gpsc=new SimulaGps(puertos[0]).getGps();
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
		
		SimpleDateFormat FechaFich=new SimpleDateFormat("yyyyMMddHHmm");
		String NombFich="Prueba"+FechaFich.format(new Date());
		
		final JTextField jtfNombreFich=new JTextField(NombFich);
		
		jbSalvar.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
//				gpsc.getBufferRutaEspacial().actualizaCoordenadasLocales();
//				gpsc.getBufferRutaTemporal().actualizaCoordenadasLocales();
				gpsc.stopRutaAndSave("Rutas/"+jtfNombreFich.getText()+".gps");
			}
		}
		);
		
		{ 
			JPanel jpSur=new JPanel();
			jpSur.add(jbSalvar);
			final JLabel jlNumPtos = new JLabel("Puntos en Buffer Espacial=###");
			jpSur.add(jlNumPtos);
			final JLabel jlNumPtosT = new JLabel("Puntos en Buffer Temporal=###");
			jpSur.add(jlNumPtosT);
			ventanaPrincipal.getContentPane().add(jpSur,BorderLayout.PAGE_END);
			gpsc.addGpsEventListener(new GpsEventListener() {
				public void handleGpsEvent(GpsEvent ev) {
					if(ev!=null) {
						jlNumPtos.setText(String.format("Puntos en Buffer Espacial=%d",gpsc.getBufferEspacial().getNumPuntos()));
						jlNumPtosT.setText(String.format("Puntos en Buffer Temporal=%d",gpsc.getBufferTemporal().getNumPuntos()));
					}
				}
			});
			gpsc.addGpsEventListener(new GpsEventListener() {
				public void handleGpsEvent(GpsEvent ev) {
					if(ev!=null)
						jlNumPtos.setText(String.format("Puntos en Buffer Espacial=%d",gpsc.getBufferEspacial().getNumPuntos()));
				}
			});
		}
		
		
		//ventanaPrincipal.setSize(new Dimension(800,400));
		ventanaPrincipal.pack();
		ventanaPrincipal.setVisible(true);
		
		
	}

	/**
	 * @return el soloEspaciales
	 */
	public boolean isSoloEspaciales() {
		return jcbSoloEspa.isSelected();
	}

	/**
	 * @param soloEspaciales el soloEspaciales a establecer
	 */
	public void setSoloEspaciales(boolean soloEspaciales) {
		jcbSoloEspa.setSelected(soloEspaciales);
	}

}
