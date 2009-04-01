package sibtra.controlcarro;

import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.JProgressBar;
import javax.swing.JSpinner;
import javax.swing.SpinnerNumberModel;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import sibtra.util.LabelDatoFormato;
import sibtra.util.PanelDatos;

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

	public PanelCarro(ControlCarro cc) {
		super();
		contCarro=cc;

		setLayout(new GridLayout(0,4)); //empezamos con 4 columnas

		//angulo volante
		añadeAPanel(new LabelDatoFormato("##.##º",ControlCarro.class,"getAnguloVolanteGrados","%5.2f º")
		, "Ángulo Volante");		
		//Consigna Volante en grados
		añadeAPanel(new LabelDatoFormato("##.## º",ControlCarro.class,"getConsignaAnguloVolanteGrados","%5.2f º")
		, "Consg Volante º");
//		//consigna volante en cuentas 	
//		añadeAPanel(new LabelDatoFormato("######",ControlCarro.class,"getConsignaVolante","%10d")
//		, "Consg Volante");
		{// spiner fijar consigna volante en grados
			jspMConsignaVolante=new SpinnerNumberModel(0.0,-45.0,45.0,0.5);
			JSpinner jspcv=new JSpinner(jspMConsignaVolante);
			añadeAPanel(jspcv, "Consg Volant º");
			jspcv.setEnabled(true);
		}
		{// Boton aplicar consigna volante en grados
			jbAplicaConsignaVolante=new JButton("Aplicar Consg Volant");
			añadeAPanel(jbAplicaConsignaVolante, "Consg Volant");
			jbAplicaConsignaVolante.addActionListener(this);
			jbAplicaConsignaVolante.setEnabled(true);
		}

		//Velocidad en m/s
		añadeAPanel(new LabelDatoFormato("##.##",ControlCarro.class,"getVelocidadMS","%5.2f")
		, "Vel. m/s");
		//Consigna Velocidad
		añadeAPanel(new LabelDatoFormato("####",ControlCarro.class,"getConsignaAvanceMS","%5.2f")
		, "Consg Velo");
		{// spiner consigna velocidad en m/s
			jspMConsignaVelocidad=new SpinnerNumberModel(1.0,0.0,6.0,0.1);
			JSpinner jspcv=new JSpinner(jspMConsignaVelocidad);
			añadeAPanel(jspcv, "Consg Veloc m/s");
			jspcv.setEnabled(true);
		}
		{// Boton aplicar
			jbAplicaConsignaVelocidad=new JButton("Aplicar Consigna");
			añadeAPanel(jbAplicaConsignaVelocidad, "Consg Velocidad");
			jbAplicaConsignaVelocidad.addActionListener(this);
			jbAplicaConsignaVelocidad.setEnabled(true);
		}

		//Bytes
		añadeAPanel(new LabelDatoFormato("######",ControlCarro.class,"getBytes","%10d")
		, "Bytes");
		//cuenta volante
		añadeAPanel(new LabelDatoFormato("######",ControlCarro.class,"getVolante","%10d")
		, "Cuenta Volante");
		//Avance
		añadeAPanel(new LabelDatoFormato("######",ControlCarro.class,"getComando","%10d")
		, "Comando");

		{//barra progreso comando velocidad
			jBarraComVel=new JProgressBar(0,maxComVelocidad);
			añadeAPanel(jBarraComVel, "Comando Velocidad");
			jBarraComVel.setOrientation(JProgressBar.HORIZONTAL);
			jBarraComVel.setValue(0);
		}
		
		//Para masFrena
		{// spiner fijar tiempo Frena
			jspTiempoFrena=new SpinnerNumberModel(20,0,255,1);
			JSpinner jspcv=new JSpinner(jspTiempoFrena);
			añadeAPanel(jspcv, "Tiempo Frena");
			jspcv.setEnabled(true);
		}
		{// spiner fijar valor frena
			jspValorFrena=new SpinnerNumberModel(90,0,255,1);
			JSpinner jspcv=new JSpinner(jspValorFrena);
			añadeAPanel(jspcv, "Valor Frena");
			jspcv.setEnabled(true);
		}
		{// Boton aplicar consigna volante en grados
			jbAplicaFreno=new JButton("Aplicar Freno");
			añadeAPanel(jbAplicaFreno, "Freno");
			jbAplicaFreno.addActionListener(this);
			jbAplicaFreno.setEnabled(true);
		}

		//Para menosFrena
		{// spiner fijar tiempo Frena
			jspTiempoDesFrena=new SpinnerNumberModel(20,0,255,1);
			JSpinner jspcv=new JSpinner(jspTiempoDesFrena);
			añadeAPanel(jspcv, "Tiempo Desfrena");
			jspcv.setEnabled(true);
		}
		{// spiner fijar valor frena
			jspValorDesFrena=new SpinnerNumberModel(90,0,255,1);
			JSpinner jspcv=new JSpinner(jspValorDesFrena);
			añadeAPanel(jspcv, "Valor Desfrena");
			jspcv.setEnabled(true);
		}
		{// Boton aplicar desfreno
			jbAplicaDesFreno=new JButton("Aplicar Desfreno");
			añadeAPanel(jbAplicaDesFreno, "Desfreno");
			jbAplicaDesFreno.addActionListener(this);
			jbAplicaDesFreno.setEnabled(true);
		}
		{// Boton desfrena total
			jbDesfrena=new JButton("Desfrena");
			añadeAPanel(jbDesfrena, "Desfreno");
			jbDesfrena.addActionListener(this);
			jbDesfrena.setEnabled(true);
		}
//		Hay lazo cerrado no tienen sentido fijar ahora el avance
//		{// comando avance
//		jspMAvance=new SpinnerNumberModel(0,0,255,5);
//		JSpinner jspcv=new JSpinner(jspMAvance);
//		jspcv.setBorder(BorderFactory.createTitledBorder(
//		blackline, "Avance"));
//		jspMAvance.addChangeListener(this);
//		add(jspcv);
//		}

		//No es necesario el moando calculado se ve en "Consigna Velocidad"
//		{ //Consigna de velocidad calculada para cada instante
//		jlConsigVelCalc=jla=new JLabel("##.##");
//		jla.setBorder(BorderFactory.createTitledBorder(
//		blackline, "Vel. m/s"));
//		jla.setFont(Grande);
//		jla.setHorizontalAlignment(JLabel.CENTER);
//		jla.setEnabled(false);
//		add(jla);
//		}
		//Alarma Freno
		añadeAPanel(new LabelDatoFormato("#",ControlCarro.class,"getFreno","%10d")
		, "Alarm. Freno");

		//Alarma Freno
		añadeAPanel(new LabelDatoFormato("#",ControlCarro.class,"getDesfreno","%10d")
		, "Alar. Desfreno");

		{// spiner fijar valor FactorFreno
			jspValorFactorFrena=new SpinnerNumberModel(contCarro.FactorFreno,0,50,0.1);
			JSpinner jspcv=new JSpinner(jspValorFactorFrena);
			añadeAPanel(jspcv, "Valor Desfrena");
			jspValorFactorFrena.addChangeListener(this);
			jspcv.setEnabled(true);
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
	}

//	Si ponemos spiner para el avance
//	public void stateChanged(ChangeEvent ev) {
//		if(ev.getSource()==jspMAvance){
//			contCarro.Avanza(jspMAvance.getNumber().intValue());
//		}
//	}


}
