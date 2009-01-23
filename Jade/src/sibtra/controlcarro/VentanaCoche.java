package sibtra.controlcarro;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Font;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Vector;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSpinner;
import javax.swing.SpinnerNumberModel;
import javax.swing.SwingUtilities;
import javax.swing.border.Border;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import sibtra.util.EligeSerial;
import sibtra.util.LabelDato;
import sibtra.util.LabelDatoFormato;

/** 
 * Ventana para la monitorización de la información recibida del coche a través
 * del {@link ControlCarro}
 * @author alberto,jonay
 *
 */

@SuppressWarnings("serial")
public class VentanaCoche extends JFrame implements ActionListener, ChangeListener, Runnable {

	private Font Grande;
	private Border blackline = BorderFactory.createLineBorder(Color.black);
	private JPanel jpCentro;
	private Vector<LabelDato> vecLabels=new Vector<LabelDato>();

	private ControlCarro contCarro;
	private SpinnerNumberModel jspMConsignaVolante;
	private JButton jbAplicaConsignaVolante;
	/** para saber si se están recibiendo paquetes del coche */
	private int cuentaBytes;
	private SpinnerNumberModel jspMConsignaVelocidad;
	private JButton jbAplicaConsignaVelocidad;
	private SpinnerNumberModel jspMAvance;
	private Thread ThreadPanel;
	
	public void run() {
		while (true){
			setEnabled(true);
			actualiza();
			repinta();
			try{Thread.sleep(500);} catch (Exception e) {}	
		}
	}

	public VentanaCoche(ControlCarro cc) {
		super("Control Carro");
		if(cc==null) 
			throw new IllegalArgumentException("Control de carro pasado no puede ser null");

		contCarro=cc;

		jpCentro=new JPanel(new GridLayout(0,4)); //empezamos con 4 columnas
		setLayout(new BorderLayout());

		//cuenta bytes
		LabelDato lda=new LabelDatoFormato("######",ControlCarro.class,"getBytes","%10d");
		Grande = lda.getFont().deriveFont(20.0f);
		//SE AÑADE AL MAS ABAJO
		
		//angulo volante
		añadeLabelDatos(new LabelDatoFormato("##.##º",ControlCarro.class,"getAnguloVolanteGrados","%5.2f º")
		, "Ángulo Volante");		
		//Consigna Volante en grados
		añadeLabelDatos(new LabelDatoFormato("##.## º",ControlCarro.class,"getConsignaAnguloVolanteGrados","%5.2f º")
		, "Consg Volante º");
//		//consigna volante en cuentas 	
//		añadeLabelDatos(new LabelDatoFormato("######",ControlCarro.class,"getConsignaVolante","%10d")
//		, "Consg Volante");
		{// spiner fijar consigna volante en grados
			jspMConsignaVolante=new SpinnerNumberModel(0.0,-45.0,45.0,0.5);
			JSpinner jspcv=new JSpinner(jspMConsignaVolante);
			jspcv.setBorder(BorderFactory.createTitledBorder(
					blackline, "Consg Volant º"));
			jpCentro.add(jspcv);
		}
		{// Boton aplicar consigna volante en grados
			jbAplicaConsignaVolante=new JButton("Aplicar Consg Volant");
			jbAplicaConsignaVolante.setBorder(BorderFactory.createTitledBorder(
					blackline, "Consg Volant"));
			jbAplicaConsignaVolante.addActionListener(this);
			jpCentro.add(jbAplicaConsignaVolante);
		}

		//Velocidad en m/s
		añadeLabelDatos(new LabelDatoFormato("##.##",ControlCarro.class,"getVelocidadMS","%5.2f")
		, "Vel. m/s");
		//Consigna Velocidad
		añadeLabelDatos(new LabelDatoFormato("####",ControlCarro.class,"getConsignaVelocidad","%10d")
		, "Consg Velo");
		{// spiner consigna velocidad en m/s
			jspMConsignaVelocidad=new SpinnerNumberModel(0.0,0.0,6.0,0.1);
			JSpinner jspcv=new JSpinner(jspMConsignaVelocidad);
			jspcv.setBorder(BorderFactory.createTitledBorder(
					blackline, "Consg Veloc m/s"));
			jpCentro.add(jspcv);
		}
		{// Boton aplicar
			jbAplicaConsignaVelocidad=new JButton("Aplicar Consigna");
			jbAplicaConsignaVelocidad.setBorder(BorderFactory.createTitledBorder(
					blackline, "Consg Velocidad"));
			jbAplicaConsignaVelocidad.addActionListener(this);
			jpCentro.add(jbAplicaConsignaVelocidad);
		}

		//Bytes
		añadeLabelDatos(lda, "Bytes");
		//cuenta volante
		añadeLabelDatos(new LabelDatoFormato("######",ControlCarro.class,"getVolante","%10d")
		, "Cuenta Volante");
		//Avance
		añadeLabelDatos(new LabelDatoFormato("######",ControlCarro.class,"getAvance","%10d")
		, "Avance");

// Hay lazo cerrado no tienen sentido fijar ahora el avance
//		{// comando avance
//			jspMAvance=new SpinnerNumberModel(0,0,255,5);
//			JSpinner jspcv=new JSpinner(jspMAvance);
//			jspcv.setBorder(BorderFactory.createTitledBorder(
//					blackline, "Avance"));
//			jspMAvance.addChangeListener(this);
//			jpCentro.add(jspcv);
//		}

		//No es necesario el moando calculado se ve en "Consigna Velocidad"
//		{ //Consigna de velocidad calculada para cada instante
//		jlConsigVelCalc=jla=new JLabel("##.##");
//		jla.setBorder(BorderFactory.createTitledBorder(
//		blackline, "Vel. m/s"));
//		jla.setFont(Grande);
//		jla.setHorizontalAlignment(JLabel.CENTER);
//		jla.setEnabled(false);
//		jpCentro.add(jla);
//		}

		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		add(jpCentro,BorderLayout.CENTER);
		pack();
		setVisible(true);
		ThreadPanel = new Thread(this);
		ThreadPanel.start();

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

	/** atendemos pulsación boton aplicar comando volante */
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
	

	public void stateChanged(ChangeEvent ev) {
		if(ev.getSource()==jspMAvance){
			contCarro.Avanza(jspMAvance.getNumber().intValue());
		}
	}

	/** Actualiza campos con datos del {@link ControlCarro} */
	public void actualiza() {
		boolean hayDato=contCarro.getBytes()!=cuentaBytes;
		cuentaBytes=contCarro.getBytes();
		//atualizamos etiquetas en array
		for(int i=0; i<vecLabels.size(); i++)
			vecLabels.elementAt(i).Actualiza(contCarro,hayDato);
	}
	
	/** programa la actualizacion de la ventana */
	public void repinta() {
		SwingUtilities.invokeLater(new Runnable() {
			public void run() {
				repaint();						
			}
		});		
	}

	public static void main(String[] args) {
		String[] puertos;
		if(args==null || args.length<1) {
			//no se han pasado argumentos, pedimos los puertos interactivamente
			String[] titulos={"Carro"};			
			puertos=new EligeSerial(titulos).getPuertos();
			if(puertos==null) {
				System.err.println("No se asignaron los puertos seriales");
				System.exit(1);
			}
		} else puertos=args;
		
		ControlCarro contCarro=new ControlCarro(puertos[0]);

		
		VentanaCoche pc = new VentanaCoche(contCarro);
		
		
		while (true){
		
			try{Thread.sleep(500);} catch (Exception e) {}	
		}
	}


	
}
