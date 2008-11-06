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

import sibtra.util.EligeSerial;

@SuppressWarnings("serial")
public class PanelCoche extends JPanel implements ActionListener {
	
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

	public PanelCoche(ControlCarro cc) {
		if(cc==null) 
			throw new IllegalArgumentException("Contorl de carro pasado no puede ser null");
		
		contCarro=cc;
		
		JPanel jpCentro=new JPanel(new GridLayout(0,3)); //empezamos con 3 columnas
		setLayout(new BorderLayout());
		add(jpCentro,BorderLayout.CENTER);

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

		{ //Avance
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
			jbAplicaConsignaVolante=new JButton("Aplicar Consigna");
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

	}

	/** atendemos pulsación boton aplicar comando volante */
	public void actionPerformed(ActionEvent ev) {
		if(ev.getSource()==jbAplicaConsignaVolante) {
			double angDeseado=Math.toRadians(jspMConsignaVolante.getNumber().doubleValue());
			contCarro.setAnguloVolante(angDeseado);
		}
	}
	
	/** Actualiza campos con datos del {@link ControlCarro} */
	public void actualiza() {
		Boolean estado=contCarro.getBytes()!=cuentaBytes;
		cuentaBytes=contCarro.getBytes();

		jlCuentaBytes.setText(String.format("%10d", contCarro.getBytes()));
		jlCuentaVolante.setText(String.format("%10d", contCarro.getVolante()));
		jlAvance.setText(String.format("%10d", contCarro.getAvance()));
		jlAnguloVolante.setText(String.format("%5.2f", Math.toDegrees(contCarro.getAnguloVolante())));
		jlConVelo.setText(String.format("%10d", contCarro.getConsignaVelocidad()));
		jlConVol.setText(String.format("%10d", contCarro.getConsignaVolante()));
		jlConAngVolante.setText(String.format("%5.2f", Math.toDegrees(contCarro.getConsignaAnguloVolante())));
		
		//fijamos estado
		jlCuentaVolante.setEnabled(estado);
		jlAnguloVolante.setEnabled(estado);
		jlCuentaBytes.setEnabled(estado);
		jlAvance.setEnabled(estado);
		jlConVelo.setEnabled(estado);
		jlConVol.setEnabled(estado);
		jbAplicaConsignaVolante.setEnabled(estado);
		jlConAngVolante.setEnabled(estado);
		

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
			String[] titulos={"GPS"};			
			puertos=new EligeSerial(titulos).getPuertos();
			if(puertos==null) {
				System.err.println("No se asignaron los puertos seriales");
				System.exit(1);
			}
		} else puertos=args;
		
		ControlCarro contCarro=new ControlCarro(puertos[0]);

		JFrame ventana=new JFrame("Panel Coche");
		ventana.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		PanelCoche pc = new PanelCoche(contCarro);
		ventana.add(pc,BorderLayout.CENTER);
		ventana.pack();
		ventana.setVisible(true);
		
		while (true){
			pc.setEnabled(true);
			pc.actualiza();
			pc.repinta();
			try{Thread.sleep(500);} catch (Exception e) {}	
		}
	}


	
}
