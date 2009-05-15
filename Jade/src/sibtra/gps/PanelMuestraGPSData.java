/**
 * 
 */
package sibtra.gps;

import java.awt.BorderLayout;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.text.SimpleDateFormat;
import java.util.Date;

import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;

import sibtra.util.EligeSerial;
import sibtra.util.LabelDato;
import sibtra.util.LabelDatoFormato;
import sibtra.util.PanelDatos;
import Jama.Matrix;

/**
 * {@link JPanel} para mostrar los datos más significativos de un GPSData.
 * @author alberto
 *
 */
public class PanelMuestraGPSData extends PanelDatos implements GpsEventListener {

	/** Para seleccionar se sólo se muestran datos espaciales */
	private JCheckBox jcbSoloEspa;
	/** Hora del ultimo dato presentado */
	private Object horaUltima="";
	
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
		super();
		//empezamos con 3 columnas
		setPanelPorDefecto(new JPanel(new GridLayout(0,3)));
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
		añadeAPanel(lda,"Hora");

		//Latitud
		añadeAPanel(new LabelDatoFormato("+?? ??.?????",GPSData.class,"getLatitudText","%s")
		, "Latitud");
		//Longitud
		añadeAPanel(new LabelDatoFormato("+??? ??.?????",GPSData.class,"getLongitudText","%s")
		,"Longitud");
		//RMS
		añadeAPanel(new LabelDatoFormato("?.???",GPSData.class,"getRms","%2.3f")
		,"RMS");
		//Número satélites
		añadeAPanel(new LabelDatoFormato("?",GPSData.class,"getSatelites","%1d")
		,"Num Satelites");
		//Edad correción diferencial
		añadeAPanel(new LabelDatoFormato("??? sg",GPSData.class,"getAge","%3.0f sg")
		, "Edad Correccion");
		//Coordenadas locales por separado
		añadeAPanel(new LabelDato("+???.??") {
			public void Actualiza(Object oa,boolean hayCambio) {
				setEnabled(hayCambio);
				if(!hayCambio) return;
				Matrix cl=((GPSData)oa).getCoordLocal();
				if(cl!=null && cl.get(0, 0)<Double.MAX_VALUE) {
					setText(String.format("%3.3f", cl.get(0,0)));
				} else 
					setEnabled(false);
			}
		}
		,"X Local");
		añadeAPanel(new LabelDato("+???.??") {
			public void Actualiza(Object oa,boolean hayCambio) {
				setEnabled(hayCambio);
				if(!hayCambio) return;
				Matrix cl=((GPSData)oa).getCoordLocal();
				if(cl!=null && cl.get(0, 0)<Double.MAX_VALUE) {
					setText(String.format("%3.3f", cl.get(1,0)));
				} else 
					setEnabled(false);
			}
		}
		,"Y Local");
		añadeAPanel(new LabelDato("+???.??") {
			public void Actualiza(Object oa,boolean hayCambio) {
				setEnabled(hayCambio);
				if(!hayCambio) return;
				Matrix cl=((GPSData)oa).getCoordLocal();
				if(cl!=null && cl.get(0, 0)<Double.MAX_VALUE) {
					setText(String.format("%3.3f", cl.get(2,0)));
				} else 
					setEnabled(false);
			}
		}
		,"Z Local");
		//altura
		añadeAPanel(new LabelDatoFormato("+????.??",GPSData.class,"getAltura","%+8.2f"), "Altura");

		//Velocidad m/€s
		añadeAPanel(new LabelDatoFormato("+??.??",GPSData.class,"getVelocidad","%+6.2f")
		, "Velocidad m/s");

		añadeAPanel(new LabelDatoFormato("+??.??",GPSData.class,"getDesvLatitud","%+7.3f")
		, "Desv. Latitud");

		añadeAPanel(new LabelDatoFormato("+??.??",GPSData.class,"getDesvLongitud","%+7.3f")
		, "Des. Longitud");

		añadeAPanel(new LabelDatoFormato("+??.??",GPSData.class,"getDesvAltura","%+7.3f")
		, "Desv. Altura");
//		Yaw
		añadeAPanel(new LabelDato("+????.??") {
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
		añadeAPanel(new LabelDatoFormato("+????.??",GPSData.class,"getAngulo","%+8.2f")
		,"Angulo Calc.");
		//Diff angulos
		añadeAPanel(new LabelDato("+????.??"){
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
		add(getPanelPorDefecto(),BorderLayout.CENTER);
		jcbSoloEspa=new JCheckBox("Sólo datos espaciales");
		jcbSoloEspa.setSelected(soloEspaciales);
		add(jcbSoloEspa,BorderLayout.SOUTH);
	}
	
	public void actualizaPunto(GPSData pto) {
		if(pto!=null)
			if(pto.getHora()==null || pto.getHora().equals(horaUltima))
				pto=null;
			else
				horaUltima=pto.getHora();
		actualizaDatos(pto);
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
	}

	/**
	 * @see sibtra.gps.GpsEventListener#handleGpsEvent(sibtra.gps.GpsEvent)
	 */
	public void handleGpsEvent(GpsEvent ev) {
		//atendemos sólo un tipo de evento según el tipo
		if(ev!=null) {
			if(jcbSoloEspa.isSelected() && ev.isEspacial()) {
				actualizaPunto(ev.getNuevoPunto());
				repinta();
			}
			if(!jcbSoloEspa.isSelected() && !ev.isEspacial()) {
				actualizaPunto(ev.getNuevoPunto());				
				repinta();
			}
		}
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

}
