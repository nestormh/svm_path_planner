/**
 * 
 */
package sibtra.gps;

import java.awt.BorderLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;

import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;

/**
 * @author alberto
 *
 */
public class ExaminaFicherosRuta extends JFrame implements  ItemListener, ActionListener {

	private Ruta rutaEspacial=null, rutaTemporal=null;
	private PanelExaminaRuta per;
	
	private JFileChooser fc;
	private JLabel jlNomF;
	private JCheckBox jcbTemporal;
	private PanelMuestraRuta pmr;
	
	public ExaminaFicherosRuta(String titulo) {
		
		super(titulo);
		fc=new JFileChooser(new File("./Rutas"));
		per=new PanelExaminaRuta();
		add(per);

		{
			JPanel jpSur=new JPanel();
			jlNomF=new JLabel("Fichero: ");
			jpSur.add(jlNomF);

			jcbTemporal=new JCheckBox("Mostrar Temporal");
			jcbTemporal.addItemListener(this);
			jpSur.add(jcbTemporal);

			JButton jbFichero=new JButton("Abrir Fichero");
			jbFichero.addActionListener(this);
			jpSur.add(jbFichero);

			getContentPane().add(jpSur,BorderLayout.PAGE_END);
		}
		
		{
			//Ventana gráfica
			JFrame jfMR=new JFrame("Muestra ruta");
			pmr=new PanelMuestraRuta(null);
			jfMR.add(pmr);
			jfMR.setSize(800, 600);
			jfMR.setVisible(true);
		}

		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		pack();
		setVisible(true);
	}

	
	public void actionPerformed(ActionEvent e) {
		int devuelto=fc.showOpenDialog(this);
		if(devuelto==JFileChooser.APPROVE_OPTION) {
			boolean seCargo=false;
			File file=fc.getSelectedFile();
			try {
				ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file));
				rutaEspacial=(Ruta)ois.readObject();
				rutaTemporal=(Ruta)ois.readObject();
				ois.close();
				if (jcbTemporal.isSelected()) {
					per.setRuta(rutaTemporal);
					pmr.setRuta(rutaTemporal);
				} else {
					per.setRuta(rutaEspacial);
					pmr.setRuta(rutaEspacial);
				}
				jlNomF.setText("Fichero: "+file.getName());
			} catch (IOException ioe) {
				System.err.println("Error al abrir el fichero " + file.getName());
				System.err.println(ioe.getMessage());
			} catch (ClassNotFoundException cnfe) {
				System.err.println("Objeto leído inválido: " + cnfe.getMessage());            
			}
		}
	}

	public void itemStateChanged(ItemEvent e) {
		if(e.getStateChange()==ItemEvent.SELECTED) {
			per.setRuta(rutaTemporal);
			pmr.setRuta(rutaTemporal);
		} else {
			per.setRuta(rutaEspacial);
			pmr.setRuta(rutaEspacial);
		}
	}
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		new ExaminaFicherosRuta("Examina Fich Ruta");
	}


}
