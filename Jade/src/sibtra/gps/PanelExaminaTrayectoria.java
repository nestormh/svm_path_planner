/**
 * 
 */
package sibtra.gps;

import java.awt.event.ActionEvent;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;

import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JSpinner;
import javax.swing.SpinnerNumberModel;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import sibtra.util.PanelFlow;
import sibtra.util.PanelMuestraTrayectoria;

/**
 * Panel que permitirá examinar los puntos de una trayectoria.
 * 
 * @author alberto
 *
 */
@SuppressWarnings("serial")
public class PanelExaminaTrayectoria extends PanelMuestraTrayectoria implements ChangeListener {
	

	private JButton jbPrimero;
	private SpinnerNumberModel spm;
	private JSpinner jsDato;
	private JLabel jlDeMaximo;
	private JButton jbUltimo;
	private JLabel jlInfoPtoAct;
	private String cadenaInfoPto="Punto Actual: (%6.2f,%6.2f,%6.2f)  %6.2fº %4.2f m/s";
	private JLabel jlInfoTray;
	private String cadenaInfoTray=" Longitud: %7.2f m Cerrada:%s";

	public PanelExaminaTrayectoria() {
		super();
		
		//Información general de la trayectoria
		{
			PanelFlow jpInfGen=new PanelFlow();
			//Información general de la trayectoria
			jlInfoTray=new JLabel(String.format(cadenaInfoTray
					, 9999.99 //largo
					, "false" //cerrada
					));
			jlInfoTray.setEnabled(false);
			jpInfGen.add(jlInfoTray);

			//añadimo botones para cambiar punto de la trayectoria
			jbPrimero=new JButton("<<");
			jbPrimero.setToolTipText("Primero");
			jbPrimero.addActionListener(this);
			jbPrimero.setEnabled(false);
			jpInfGen.add(jbPrimero);
			
			spm=new SpinnerNumberModel(1,1,100000,1);
			spm.addChangeListener(this);
			jsDato=new JSpinner(spm);
			jsDato.setSize(150, jsDato.getHeight());
			jsDato.setEnabled(false);
			jpInfGen.add(jsDato);
			
			jlDeMaximo=new JLabel(" de ???? ");
			jpInfGen.add(jlDeMaximo);
			jlDeMaximo.setEnabled(false);
			
			jbUltimo=new JButton(">>");
			jbUltimo.setToolTipText("Ultimo");
			jbUltimo.addActionListener(this);
			jbUltimo.setEnabled(false);
			jpInfGen.add(jbUltimo);
			
			jlInfoPtoAct=new JLabel(String.format(cadenaInfoPto
					, -999.99 //x
					, -999.99 //y
					, -999.99 //z
					, -179.99 //rumbo
					, 5.99 //velocidad
					));
			jlInfoPtoAct.setEnabled(false);
			jpInfGen.add(jlInfoPtoAct);
			
			add(jpInfGen);
		}
		
	}
	
	public PanelExaminaTrayectoria(Trayectoria trIni) {
		this();
		setTrayectoria(trIni);
	}
	
	public void setTrayectoria(Trayectoria nTra) {
		super.setTrayectoria(nTra);
		boolean hayTra=(tray!=null);
		boolean hayPtos=hayTra && tray.length()>0;
		jlInfoTray.setEnabled(hayTra);
		jbPrimero.setEnabled(hayPtos);
		jsDato.setEnabled(hayPtos);
		jbUltimo.setEnabled(hayPtos);
		jlInfoPtoAct.setEnabled(hayPtos);
		jlDeMaximo.setEnabled(hayTra);
		if(hayPtos) {
			spm.setMaximum(tray.length()+1);
			spm.setValue(1);
		}
		if(hayTra) {
			jlInfoTray.setText(String.format(cadenaInfoTray
					, tray.getLargo(), tray.esCerrada()));
			jlDeMaximo.setText(" de "+tray.length()+" ");
		}
	}
	
	
	/* (non-Javadoc)
	 * @see java.awt.event.ActionListener#actionPerformed(java.awt.event.ActionEvent)
	 */
	public void actionPerformed(ActionEvent ae) {
		super.actionPerformed(ae);
		if(ae.getSource()==jbPrimero) {
			spm.setValue(1);
		} else if(ae.getSource()==jbUltimo) {
			spm.setValue(tray.length());
		} 
	}

	/* (non-Javadoc)
	 * @see javax.swing.event.ChangeListener#stateChanged(javax.swing.event.ChangeEvent)
	 */
	public void stateChanged(ChangeEvent ce) {
		if(ce.getSource()==spm) {
			int indSel=(Integer)spm.getValue()-1;
			jlInfoPtoAct.setText(String.format(cadenaInfoPto
					, tray.x[indSel] 
					, tray.y[indSel] 
					, tray.z[indSel] 
					, tray.rumbo[indSel] 
					, tray.velocidad[indSel] 
					));
			situaCoche(tray.x[indSel], tray.y[indSel], tray.rumbo[indSel]);
			repaint();
		}
	}
	
	public static void main(String[] args) {
		Ruta rutaEspacial=null;
		JFileChooser fc=new JFileChooser(new File("./Rutas"));
		do {
			int devuelto=fc.showOpenDialog(null);
			if (devuelto!=JFileChooser.APPROVE_OPTION) 
				JOptionPane.showMessageDialog(null,
						"Necesario cargar fichero de ruta",
						"Error",
						JOptionPane.ERROR_MESSAGE);
			else  {
				String fichRuta=fc.getSelectedFile().getAbsolutePath();
				try {
					File file = new File(fichRuta);
					ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file));
					rutaEspacial=(Ruta)ois.readObject();
					ois.close();
				} catch (IOException ioe) {
					System.err.println("Error al abrir el fichero " + fichRuta);
					System.err.println(ioe.getMessage());
					rutaEspacial=null;
				} catch (ClassNotFoundException cnfe) {
					System.err.println("Objeto leído inválido: " + cnfe.getMessage());
					rutaEspacial=null;
				}     
			}
		} while(rutaEspacial==null);
		
		//Tenemos fichero de ruta Creamos trayectoria
		Trayectoria tr=new Trayectoria(rutaEspacial);
		
		JFrame ventana=new JFrame("Examina Trayectoria");
		ventana.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		PanelExaminaTrayectoria pet=new PanelExaminaTrayectoria(tr);
		ventana.add(pet);
		
		ventana.pack();
		ventana.setVisible(true);

		
		
	}
}
