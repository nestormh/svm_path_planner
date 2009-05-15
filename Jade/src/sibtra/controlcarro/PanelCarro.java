package sibtra.controlcarro;

import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JProgressBar;
import javax.swing.JSpinner;
import javax.swing.SpinnerNumberModel;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import sibtra.gps.PanelExaminaRuta;
import sibtra.gps.Ruta;
import sibtra.util.LabelDatoFormato;
import sibtra.util.PanelDatos;
import sibtra.util.SpinnerDouble;
import sibtra.util.SpinnerInt;

/**
 * {@link PanelDatos} para mostrar la información del carro y permitir fijar consignas de
 * velocidad y volante.
 * @author alberto
 */
public class PanelCarro extends PanelDatos implements ActionListener, ChangeListener {

	/** Maximo comando de velocidad esperado */
	private static final int maxComVelocidad = 255;

	/** Conexión al carro*/
	private ControlCarro contCarro;

	/** Para elegir consigna para el volante */
	private SpinnerNumberModel jspMConsignaVolante;
	/** Para aplicar consigna del volante */
	private JButton jbAplicaConsignaVolante;

	/** para saber si se están recibiendo paquetes del coche */
	private int cuentaBytes=0;
	
	/** Para elegir consigna para la velocidad */
	private SpinnerNumberModel jspMConsignaVelocidad;
	/** Para aplicar consigna de velocidad */
	private JButton jbAplicaConsignaVelocidad;
//	private SpinnerNumberModel jspMAvance;
	private JProgressBar jBarraComVel;

	private SpinnerNumberModel jspTiempoFrena;

	private SpinnerNumberModel jspValorFrena;

	private JButton jbAplicaFreno;

	private SpinnerNumberModel jspTiempoDesFrena;

	private SpinnerNumberModel jspValorDesFrena;

	private JButton jbAplicaDesFreno;

	private JButton jbDesfrena;

	private SpinnerNumberModel jspValorFactorFrena;

	private JButton jbParaControl;

	public PanelCarro(ControlCarro cc) {
		super();
		contCarro=cc;

		setLayout(new GridLayout(0,3)); //empezamos con 4 columnas

		//angulo volante
		añadeAPanel(new LabelDatoFormato("##.##º",ControlCarro.class,"getAnguloVolanteGrados","%5.2f º")
		, "Ángulo Volante");		
		//Consigna Volante en grados
		añadeAPanel(new LabelDatoFormato("##.## º",ControlCarro.class,"getConsignaAnguloVolanteGrados","%5.2f º")
		, "Csg Volante º");
//		//consigna volante en cuentas 	
//		añadeAPanel(new LabelDatoFormato("######",ControlCarro.class,"getConsignaVolante","%10d")
//		, "Csg Volante");
		{// spiner fijar consigna volante en grados
			jspMConsignaVolante=new SpinnerNumberModel(0.0,-45.0,45.0,0.5);
			JSpinner jspcv=new JSpinner(jspMConsignaVolante);
			añadeAPanel(jspcv, "Csg Volant º");
		}
		{// Boton aplicar consigna volante en grados
			jbAplicaConsignaVolante=new JButton("Aplicar");
			añadeAPanel(jbAplicaConsignaVolante, "Csg Volant");
			jbAplicaConsignaVolante.addActionListener(this);
		}

		//Velocidad en m/s
		añadeAPanel(new LabelDatoFormato("##.##",ControlCarro.class,"getVelocidadMS","%5.2f")
		, "Vel. m/s");
		//Consigna Velocidad
		añadeAPanel(new LabelDatoFormato("####",ControlCarro.class,"getConsignaAvanceMS","%5.2f")
		, "Csg Velo");
		{// spiner consigna velocidad en m/s
			jspMConsignaVelocidad=new SpinnerNumberModel(1.0,0.0,6.0,0.1);
			JSpinner jspcv=new JSpinner(jspMConsignaVelocidad);
			añadeAPanel(jspcv, "Csg Veloc m/s");
		}
		{// Boton aplicar
			jbAplicaConsignaVelocidad=new JButton("Aplicar");
			añadeAPanel(jbAplicaConsignaVelocidad, "Csg Velocidad");
			jbAplicaConsignaVelocidad.addActionListener(this);
		}

		//Bytes
		añadeAPanel(new LabelDatoFormato("######",ControlCarro.class,"getBytes","%10d")
		, "Bytes");
		//cuenta volante
		añadeAPanel(new LabelDatoFormato("######",ControlCarro.class,"getVolante","%10d")
		, "Cuenta Volante");
		//Comando
		añadeAPanel(new LabelDatoFormato("######",ControlCarro.class,"getComando","%10.2f")
		, "Comando");

		{//barra progreso comando velocidad
			jBarraComVel=new JProgressBar(0,maxComVelocidad);
			añadeAPanel(jBarraComVel, "Cmd Velocidad");
			jBarraComVel.setOrientation(JProgressBar.HORIZONTAL);
			jBarraComVel.setValue(0);
		}
		
		//Para masFrena
		{// spiner fijar tiempo Frena
			jspTiempoFrena=new SpinnerNumberModel(20,0,255,1);
			JSpinner jspcv=new JSpinner(jspTiempoFrena);
			añadeAPanel(jspcv, "Timp Frena");
		}
		{// spiner fijar valor frena
			jspValorFrena=new SpinnerNumberModel(90,0,255,1);
			JSpinner jspcv=new JSpinner(jspValorFrena);
			añadeAPanel(jspcv, "Valor Frena");
		}
		{// Boton aplicar consigna volante en grados
			jbAplicaFreno=new JButton("Aplicar");
			añadeAPanel(jbAplicaFreno, "Freno");
			jbAplicaFreno.addActionListener(this);
		}

//		añadeAPanel(new JLabel("VERDINO"), "");
		//Para menosFrena
		{// spiner fijar tiempo Frena
			jspTiempoDesFrena=new SpinnerNumberModel(20,0,255,1);
			JSpinner jspcv=new JSpinner(jspTiempoDesFrena);
			añadeAPanel(jspcv, "Timp Desfr");
		}
		{// spiner fijar valor frena
			jspValorDesFrena=new SpinnerNumberModel(90,0,255,1);
			JSpinner jspcv=new JSpinner(jspValorDesFrena);
			añadeAPanel(jspcv, "Valor Desfr");
		}
		{// Boton aplicar desfreno
			jbAplicaDesFreno=new JButton("Aplicar");
			añadeAPanel(jbAplicaDesFreno, "Desfreno");
			jbAplicaDesFreno.addActionListener(this);
		}
		{// Boton desfrena total
			jbDesfrena=new JButton("TOTAL");
			añadeAPanel(jbDesfrena, "Desfreno");
			jbDesfrena.addActionListener(this);
		}

		//Alarma Freno
		añadeAPanel(new LabelDatoFormato("#",ControlCarro.class,"getFreno","%10d")
		, "Alarm. Freno");

		//Alarma Freno
		añadeAPanel(new LabelDatoFormato("#",ControlCarro.class,"getDesfreno","%10d")
		, "Alar. Desfreno");

		añadeAPanel(new SpinnerDouble(contCarro,"setFactorFreno",0,50,0.1), "Fact Freno");
		añadeAPanel(new SpinnerInt(contCarro,"setMaxIncremento",0,255*2,1), "Max Inc");
		añadeAPanel(new SpinnerDouble(contCarro,"setKPAvance",0,50,0.1), "KP Avance");
		añadeAPanel(new SpinnerDouble(contCarro,"setKIAvance",0,50,0.1), "KI Avance");
		añadeAPanel(new SpinnerDouble(contCarro,"setKDAvance",0,50,0.1), "KD Avance");

		{// Boton parar el control
			jbParaControl=new JButton("PARAR");
			añadeAPanel(jbParaControl, "Control");
			jbParaControl.addActionListener(this);
		}

	}
	
	/** atendemos pulsaciones de los botones para aplicar consignas */
	public void actionPerformed(ActionEvent ev) {
		if(ev.getSource()==jbAplicaConsignaVolante) {
			double angDeseado=Math.toRadians(jspMConsignaVolante.getNumber().doubleValue());
			contCarro.setAnguloVolante(angDeseado);
		}
		if(ev.getSource()==jbAplicaConsignaVelocidad) {
			double velDeseado=jspMConsignaVelocidad.getNumber().doubleValue();
			contCarro.setConsignaAvanceMS(velDeseado);
		}
		if(ev.getSource()==jbAplicaFreno) {
			contCarro.masFrena(jspValorFrena.getNumber().intValue(), jspTiempoFrena.getNumber().intValue());
		}
		if(ev.getSource()==jbAplicaDesFreno) {
			contCarro.menosFrena(jspValorDesFrena.getNumber().intValue(), jspTiempoDesFrena.getNumber().intValue());
		}
		if(ev.getSource()==jbDesfrena) {
			contCarro.DesFrena(255);
		}
		if(ev.getSource()==jbParaControl) {
			contCarro.stopControlVel();
			jbParaControl.setEnabled(false);
		}
	}

	public void stateChanged(ChangeEvent ce) {
		if(ce.getSource()==jspValorFactorFrena) {
			contCarro.FactorFreno=jspValorFactorFrena.getNumber().doubleValue();
		}
	}

	/** Actualiza campos con datos del {@link ControlCarro} */
	public void actualizaCarro() {
		boolean hayDato=contCarro.getBytes()!=cuentaBytes;
		cuentaBytes=contCarro.getBytes();
		if(hayDato)
			actualizaDatos(contCarro);
		else
			actualizaDatos(null);
		jBarraComVel.setValue(contCarro.getComandoVelocidad());
		if(contCarro.isControlando())
			jbParaControl.setEnabled(true);
		else
			jbParaControl.setEnabled(false);
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {

		ControlCarro cc=new ControlCarro("/dev/null");
		PanelCarro pcr=new PanelCarro(cc);
		JFrame ventana=new JFrame("Panel Carro");
		ventana.add(pcr);
		ventana.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		ventana.pack();
		ventana.setVisible(true);
	}


}
