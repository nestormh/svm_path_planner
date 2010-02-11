/**
 * 
 */
package sibtra.gps;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;

import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

/**
 * @author alberto
 *
 */
public class ExaminaFicherosRuta extends JFrame implements  ItemListener, ActionListener, ChangeListener {

	private Ruta rutaEspacial=null, rutaTemporal=null;
	private PanelExaminaRuta per;
	
	private JFileChooser fc;
	private JLabel jlNomF;
	private JCheckBox jcbTemporal;
	private PanelMuestraRuta pmr;
	private JCheckBox jcbMarcarDM;
	
	public ExaminaFicherosRuta(String titulo) {
		
		super(titulo);
		fc=new JFileChooser(new File("./Rutas"));
		JPanel cp=(JPanel)getContentPane();
		cp.setLayout(new BoxLayout(cp,BoxLayout.PAGE_AXIS));
		pmr=new PanelMuestraRuta(null);
		cp.add(pmr);
		per=new PanelExaminaRuta();
		cp.add(per);

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

			jcbMarcarDM=new JCheckBox("Marcar usados DM");
			jcbMarcarDM.addItemListener(this);
			jpSur.add(jcbMarcarDM);

			cp.add(jpSur);
		}
		
		per.jsDato.addChangeListener(this);
		
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		pack();
		setVisible(true);
		setBounds(0, 384, 1024, 742);
	}
        public Ruta getRutaTemporal(){
            return rutaTemporal;
        }
        public Ruta getRutaEspacial(){
            return rutaEspacial;
        }
	
	public void actionPerformed(ActionEvent e) {
		int devuelto=fc.showOpenDialog(this);
		if(devuelto==JFileChooser.APPROVE_OPTION) {
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
		if(e.getSource()==jcbTemporal)
			if(e.getStateChange()==ItemEvent.SELECTED) {
				per.setRuta(rutaTemporal);
				pmr.setRuta(rutaTemporal);
				rutaTemporal.getDesviacionM();
			} else {
				per.setRuta(rutaEspacial);
				pmr.setRuta(rutaEspacial);
				rutaEspacial.getDesviacionM();
			}
		if(e.getSource()==jcbMarcarDM)
			if(e.getStateChange()==ItemEvent.SELECTED) {
				pmr.setMarcados(per.ruta.indiceConsideradosDM);
				pmr.actualiza();
			} else { 
				pmr.setMarcados(null);
				pmr.actualiza();
			}
	}
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		new ExaminaFicherosRuta("Examina Fich Ruta");
	}


	public void stateChanged(ChangeEvent arg0) {
		GPSData npto=per.ruta.getPunto((Integer)per.jsDato.getValue());
		pmr.nuevoPunto(npto);
	}


}
