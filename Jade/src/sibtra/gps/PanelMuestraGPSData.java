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
import java.util.Vector;

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
import sibtra.util.LabelDato;
import sibtra.util.LabelDatoFormato;
import Jama.Matrix;

/**
 * {@link JPanel} para mostrar los datos más significativos de un GPSData.
 * @author alberto
 *
 */
public class PanelMuestraGPSData extends JPanel implements GpsEventListener {

	private Font Grande;
	private Border blackline = BorderFactory.createLineBorder(Color.black);
	private JPanel jpCentro;
	private JCheckBox jcbSoloEspa;
	
	private Vector<LabelDato> vecLabels=new Vector<LabelDato>();
;
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
		jpCentro=new JPanel(new GridLayout(0,3)); //empezamos con 3 columnas
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
		//hora
		LabelDato lda=new LabelDatoFormato("??:??:??.??",GPSData.class,"getHora","%s");
		Grande = lda.getFont().deriveFont(20.0f);
		añadeLabelDatos(lda,"Hora");

		//Latitud
		añadeLabelDatos(new LabelDatoFormato("+?? ??.?????",GPSData.class,"getLatitudText","%s")
		, "Latitud");
		//Longitud
		añadeLabelDatos(new LabelDatoFormato("+??? ??.?????",GPSData.class,"getLongitudText","%s")
		,"Longitud");
		//RMS
		añadeLabelDatos(new LabelDatoFormato("?.?",GPSData.class,"getRms","%2.1f")
		,"RMS");
		//Número satélites
		añadeLabelDatos(new LabelDatoFormato("?",GPSData.class,"getSatelites","%1d")
		,"Num Satelites");
		//Edad correción diferencial
		añadeLabelDatos(new LabelDatoFormato("??? sg",GPSData.class,"getAge","%3.0f sg")
		, "Edad Correccion");
		//Coordenadas locales
		añadeLabelDatos(new LabelDato("(???.??, ???.??, ???.??)") {
			public void Actualiza(Object oa,boolean hayCambio) {
				setEnabled(hayCambio);
				if(!hayCambio) return;
				Matrix cl=((GPSData)oa).getCoordLocal();
				if(cl!=null && cl.get(0, 0)<Double.MAX_VALUE) {
					setText(String.format("(%3.3f %3.3f %3.3f)"
							, cl.get(0,0), cl.get(1,0), cl.get(2,0)));
				} else 
					setEnabled(false);
			}
		}
		,"Locales");
		//altura
		añadeLabelDatos(new LabelDatoFormato("+????.??",GPSData.class,"getAltura","%+8.2f"), "Altura");

		//Velocidad m/€s
		añadeLabelDatos(new LabelDatoFormato("+??.??",GPSData.class,"getVelocidad","%+6.2f")
		, "Velocidad m/s");
//		Yaw
		añadeLabelDatos(new LabelDato("+????.??") {
			public void Actualiza(Object oa, boolean hayCambio) {
				setEnabled(hayCambio);
				if(!hayCambio) return;
				GPSData pto=(GPSData)oa;
				if(pto.getAngulosIMU()!=null) {
					setText(String.format("%+8.2f", pto.getAngulosIMU().getYaw()));
				} else {
					setEnabled(false);
				}
			}
		}
		,"Yaw IMU");
		//Angulo calculado
		añadeLabelDatos(new LabelDatoFormato("+????.??",GPSData.class,"getAngulo","%+8.2f")
		,"Angulo Calc.");
		//Diff angulos
		añadeLabelDatos(new LabelDato("+????.??"){
			public void Actualiza(Object oa, boolean hayCambio) {
				setEnabled(hayCambio);
				if(!hayCambio) return;
				GPSData pto=(GPSData)oa;
				if(pto.getAngulosIMU()!=null) {
					setText(String.format("%+8.2f", 
							Math.toDegrees(pto.getAngulo())-pto.getAngulosIMU().getYaw()));
				} else {
					setEnabled(false);
				}
			}
		}
		, "D. Magnetica");
		add(jpCentro,BorderLayout.CENTER);
		jcbSoloEspa=new JCheckBox("Sólo datos espaciales");
		jcbSoloEspa.setSelected(soloEspaciales);
		add(jcbSoloEspa,BorderLayout.SOUTH);
	}

	/**
	 * Funcion para añadir etiqueta con todas las configuraciones por defecto
	 * @param lda etiqueta a añadir
	 * @param Titulo titulo adjunto
	 */
	private void añadeLabelDatos(LabelDato lda,String Titulo) {
		vecLabels.add(lda);
		lda.setBorder(BorderFactory.createTitledBorder(
				blackline, Titulo));
		lda.setFont(Grande);
		lda.setHorizontalAlignment(JLabel.CENTER);
		lda.setEnabled(false);
		jpCentro.add(lda);
		
	}
	
	public void actualizaPunto(GPSData pto) {
		boolean hayDato=(pto!=null);
		//atualizamos etiquetas en array
		for(int i=0; i<vecLabels.size(); i++)
			vecLabels.elementAt(i).Actualiza(pto,hayDato);
			//Nuevos valores
//			Matrix ce=pto.getCoordECEF();
//			if(ce!=null && ce.get(0, 0)<Double.MAX_VALUE) {
//				jlCoordECEFx.setText(String.format("%10.3f",ce.get(0,0)));
//				jlCoordECEFy.setText(String.format("%10.3f",ce.get(1,0)));
//				jlCoordECEFz.setText(String.format("%10.3f",ce.get(2,0)));
//			} else {
//				jlCoordECEFx.setEnabled(false);
//				jlCoordECEFy.setEnabled(false);
//				jlCoordECEFz.setEnabled(false);
//			}
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
