package sibtra.controlcarro;

import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.JProgressBar;
import javax.swing.JSpinner;
import javax.swing.SpinnerNumberModel;

import sibtra.util.LabelDatoFormato;
import sibtra.util.PanelDatos;

/**
 * {@link PanelDatos} para mostrar la información del carro y permitir fijar consignas de
 * velocidad y volante.
 * @author alberto
 */
public class PanelCarro extends PanelDatos implements ActionListener {

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
			jspMConsignaVelocidad=new SpinnerNumberModel(0.0,0.0,6.0,0.1);
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
		añadeAPanel(new LabelDatoFormato("######",ControlCarro.class,"getAvance","%10d")
		, "Avance");

		{//barra progreso comando velocidad
			jBarraComVel=new JProgressBar(0,maxComVelocidad);
			añadeAPanel(jBarraComVel, "Comando Velocidad");
			jBarraComVel.setOrientation(JProgressBar.HORIZONTAL);
			jBarraComVel.setValue(0);
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
