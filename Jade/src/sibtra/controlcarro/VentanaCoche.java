package sibtra.controlcarro;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Font;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

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

/** 
 * Ventana para la monitorización de la información recibida del coche a través
 * del {@link ControlCarro}
 * @author alberto,jonay
 *
 */

@SuppressWarnings("serial")
public class VentanaCoche extends JFrame implements ActionListener, ChangeListener, Runnable {
	
	private JLabel jlCuentaVolante;
	private ControlCarro contCarro;
	private JLabel jlAnguloVolante;
	private JLabel jlCuentaBytes;
	private JLabel jlAvance;
	private JLabel jlConVelo;
	private JLabel jlConVol;
	private SpinnerNumberModel jspMConsignaVolante;
	private JButton jbAplicaConsignaVolante;
	private JLabel jlConAngVolante;
	/** para saber si se están recibiendo paquetes del coche */
	private int cuentaBytes;
	private JLabel jlVelMS;
	private SpinnerNumberModel jspMConsignaVelocidad;
	private JButton jbAplicaConsignaVelocidad;
	private SpinnerNumberModel jspMAvance;
	private Thread ThreadPanel;
        private JLabel jlConsigVelCalc;
	
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
		
		JPanel jpCentro=new JPanel(new GridLayout(0,3)); //empezamos con 3 columnas
		setLayout(new BorderLayout());

		Border blackline = BorderFactory.createLineBorder(Color.black);
		Font Grande;
		JLabel jla; //variable para poner JLable actual
		{ //cuenta bytes
			jlCuentaBytes=jla=new JLabel("######");
		    Grande = jla.getFont().deriveFont(20.0f);
			jla.setBorder(BorderFactory.createTitledBorder(
				       blackline, "Bytes"));
		    jla.setFont(Grande);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
			jpCentro.add(jla);
		}

		{ //cuenta volante
			jlCuentaVolante=jla=new JLabel("######");
			jla.setBorder(BorderFactory.createTitledBorder(
				       blackline, "Cuenta Volante"));
		    jla.setFont(Grande);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
			jpCentro.add(jla);
		}

		{ //angulo volante
			jlAnguloVolante=jla=new JLabel("###.## º");
			jla.setBorder(BorderFactory.createTitledBorder(
				       blackline, "Ángulo Volante"));
		    jla.setFont(Grande);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
			jpCentro.add(jla);
		}

		{ //Avance
			jlAvance=jla=new JLabel("######");
			jla.setBorder(BorderFactory.createTitledBorder(
				       blackline, "Avance"));
		    jla.setFont(Grande);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
			jpCentro.add(jla);
		}

		{ //Consigna Velocidad
			jlConVelo=jla=new JLabel("######");
			jla.setBorder(BorderFactory.createTitledBorder(
				       blackline, "Con. Velocidad"));
		    jla.setFont(Grande);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
			jpCentro.add(jla);
		}

		{
			jlConVol=jla=new JLabel("######");
			jla.setBorder(BorderFactory.createTitledBorder(
				       blackline, "Con. Volante"));
		    jla.setFont(Grande);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
			jpCentro.add(jla);
		}
		
		{// comando volante
			jspMConsignaVolante=new SpinnerNumberModel(0.0,-45.0,45.0,0.5);
			JSpinner jspcv=new JSpinner(jspMConsignaVolante);
			jspcv.setBorder(BorderFactory.createTitledBorder(
				       blackline, "Con. Volante"));
			jpCentro.add(jspcv);
		}

		{// Boton aplicar
			jbAplicaConsignaVolante=new JButton("Aplicar Consigna V");
			jbAplicaConsignaVolante.setBorder(BorderFactory.createTitledBorder(
				       blackline, "Con. Volante"));
			jbAplicaConsignaVolante.addActionListener(this);
			jpCentro.add(jbAplicaConsignaVolante);
		}

		{ //Consigna Volante en grados
			jlConAngVolante=jla=new JLabel("##.## º");
			jla.setBorder(BorderFactory.createTitledBorder(
				       blackline, "Con. Angulo Volante"));
		    jla.setFont(Grande);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
			jpCentro.add(jla);
		}

		{ //Velocidad en m/s
			jlVelMS=jla=new JLabel("##.##");
			jla.setBorder(BorderFactory.createTitledBorder(
				       blackline, "Vel. m/s"));
		    jla.setFont(Grande);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
			jpCentro.add(jla);
		}

		{// comando velocidad
			jspMConsignaVelocidad=new SpinnerNumberModel(0.0,0.0,6.0,0.1);
			JSpinner jspcv=new JSpinner(jspMConsignaVelocidad);
			jspcv.setBorder(BorderFactory.createTitledBorder(
				       blackline, "Con. Velocidad"));
			jpCentro.add(jspcv);
		}

		{// Boton aplicar
			jbAplicaConsignaVelocidad=new JButton("Aplicar Consigna");
			jbAplicaConsignaVelocidad.setBorder(BorderFactory.createTitledBorder(
				       blackline, "Con. Velocidad"));
			jbAplicaConsignaVelocidad.addActionListener(this);
			jpCentro.add(jbAplicaConsignaVelocidad);
		}


		{// comando avance
			jspMAvance=new SpinnerNumberModel(0,0,255,5);
			JSpinner jspcv=new JSpinner(jspMAvance);
			jspcv.setBorder(BorderFactory.createTitledBorder(
				       blackline, "Avance"));
			jspMAvance.addChangeListener(this);
			jpCentro.add(jspcv);
		}

		{ //Consigna de velocidad calculada para cada instante
			jlConsigVelCalc=jla=new JLabel("##.##");
			jla.setBorder(BorderFactory.createTitledBorder(
					blackline, "Vel. m/s"));
			jla.setFont(Grande);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
			jpCentro.add(jla);
		}

		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		add(jpCentro,BorderLayout.CENTER);
		pack();
		setVisible(true);
		ThreadPanel = new Thread(this);
		ThreadPanel.start();
		
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
		Boolean estado=contCarro.getBytes()!=cuentaBytes;
		cuentaBytes=contCarro.getBytes();

		if(estado) {
			jlCuentaBytes.setText(String.format("%10d", contCarro.getBytes()));
			jlCuentaVolante.setText(String.format("%10d", contCarro.getVolante()));
			jlAvance.setText(String.format("%10d", contCarro.getAvance()));
			jlAnguloVolante.setText(String.format("%5.2f", Math.toDegrees(contCarro.getAnguloVolante())));
			jlConVelo.setText(String.format("%10d", contCarro.getConsignaVelocidad()));
			jlConVol.setText(String.format("%10d", contCarro.getConsignaVolante()));
			jlConAngVolante.setText(String.format("%5.2f", Math.toDegrees(contCarro.getConsignaAnguloVolante())));
			jlVelMS.setText(String.format("%5.2f",contCarro.getVelocidadMS()));
		}                
		//fijamos estado
		jlCuentaVolante.setEnabled(estado);
		jlAnguloVolante.setEnabled(estado);
		jlCuentaBytes.setEnabled(estado);
		jlAvance.setEnabled(estado);
		jlConVelo.setEnabled(estado);
		jlConVol.setEnabled(estado);
		jlConAngVolante.setEnabled(estado);
		jlVelMS.setEnabled(estado);
		

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
