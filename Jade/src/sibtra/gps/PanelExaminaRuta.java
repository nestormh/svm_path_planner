/**
 * 
 */
package sibtra.gps;

import java.awt.BorderLayout;
import java.awt.Window;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSpinner;
import javax.swing.SpinnerNumberModel;
import javax.swing.WindowConstants;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;


/**
 * {@link JPanel} para examinar el contenido de una ruta.
 * @author alberto
 *
 */
public class PanelExaminaRuta extends JPanel implements ActionListener, ChangeListener {
	
	private PanelMuestraGPSData PanelPto;
	
	Ruta ruta;

	private JButton jbPrimero;
	private JButton jbAnterior;
	private JButton jbSiguiente;
	private JButton jbUltimo;

	JSpinner jsDato;

	private SpinnerNumberModel spm;

	private JLabel jlDeMaximo;

	private JLabel jlDesviaM;

	private SpinnerNumberModel spUmbral;

	public PanelExaminaRuta() {
		this(null);
	}
	
	/**
	 * 
	 */
	public PanelExaminaRuta(Ruta ra) {
		setLayout(new BorderLayout(5,5));
		
		PanelPto=new PanelMuestraGPSData();
		
		add(PanelPto,BorderLayout.CENTER);
		
		{ //controles parte inferior
			JPanel jpInf=new JPanel();
			
			jbPrimero=new JButton("<<");
			jbPrimero.setToolTipText("Primero");
			jbPrimero.addActionListener(this);
			jpInf.add(jbPrimero);
			
			jbAnterior=new JButton("<");
			jbAnterior.setToolTipText("Anterior");
			jbAnterior.addActionListener(this);
			jpInf.add(jbAnterior);
			
			spm=new SpinnerNumberModel(1,1,100000,1);
			jsDato=new JSpinner(spm);
			jsDato.setSize(150, jsDato.getHeight());
			jsDato.addChangeListener(this);
			jpInf.add(jsDato);
			
			jlDeMaximo=new JLabel(" de ???? ");
			jpInf.add(jlDeMaximo);
			
			jbSiguiente=new JButton(">");
			jbSiguiente.setToolTipText("Siguiente");
			jbSiguiente.addActionListener(this);
			jpInf.add(jbSiguiente);
			
			jbUltimo=new JButton(">>");
			jbUltimo.setToolTipText("Ultimo");
			jbUltimo.addActionListener(this);
			jpInf.add(jbUltimo);
			
			//Para la desviación magnética
			jlDesviaM=new JLabel("Desviación magnetica ##.##º con umbral ");
			jpInf.add(jlDesviaM);
			spUmbral=new SpinnerNumberModel(10.00,0,180,0.25);
			spUmbral.addChangeListener(this);
			JSpinner jspUmbral=new JSpinner(spUmbral);
			jpInf.add(jspUmbral);
			
			add(jpInf,BorderLayout.PAGE_END);
		}
		setRuta(ra);
	}
	
	public void setRuta(Ruta ra) {
		ruta=ra;
		if(ruta==null || ruta.getNumPuntos()==0) {
			jbPrimero.setEnabled(false);
			jbAnterior.setEnabled(false);
			jsDato.setEnabled(false);
			jbSiguiente.setEnabled(false);
			jbUltimo.setEnabled(false);
			jlDesviaM.setEnabled(false);
		} else {
			jbPrimero.setEnabled(true);
			jbAnterior.setEnabled(true);
			jsDato.setEnabled(true);
			jbSiguiente.setEnabled(true);
			jbUltimo.setEnabled(true);
			spm.setMaximum(ruta.getNumPuntos());
			jlDeMaximo.setText(String.format(" de %d ", ruta.getNumPuntos()));
			spm.setValue(1);
			actualizaDM();
		}
	}

	/* (non-Javadoc)
	 * @see java.awt.event.ActionListener#actionPerformed(java.awt.event.ActionEvent)
	 */
	public void actionPerformed(ActionEvent ae) {
		if(ae.getSource()==jbPrimero) {
			jsDato.setValue(1);
		} else if(ae.getSource()==jbUltimo) {
			jsDato.setValue(ruta.getNumPuntos());
		} else if (ae.getSource()==jbAnterior && (Integer)jsDato.getValue()>1) {
			jsDato.setValue((Integer)jsDato.getValue()-1);
		} else if (ae.getSource()==jbSiguiente && (Integer)jsDato.getValue()<ruta.getNumPuntos()) {
			jsDato.setValue((Integer)jsDato.getValue()+1);
		}
	}

	/* (non-Javadoc)
	 * @see javax.swing.event.ChangeListener#stateChanged(javax.swing.event.ChangeEvent)
	 */
	public void stateChanged(ChangeEvent ce) {
		if(ce.getSource()==jsDato) {
			GPSData npto=ruta.getPunto((Integer)jsDato.getValue()-1);
			PanelPto.actualizaPunto(npto);
			System.out.println(npto.getCadenaNMEA());
			
		}
		if(ce.getSource()==spUmbral) {
			actualizaDM();
		}
	}

	/** Para actializar la etiqueta con la desviación magnética */
	private void actualizaDM() {
		if(ruta==null) return;
		double dm=Math.toDegrees(ruta.getDesviacionM(Math.toRadians((Double)spUmbral.getValue())));
		jlDesviaM.setText(String.format(
				"Desviación magnetica %6.2fº (usando %d de %d) (est:%6.2fº max:%6.2fº))con umbral " 
				, dm
				,ruta.indiceConsideradosDM.size(),ruta.puntos.size()
				,Math.toDegrees(ruta.desEstDM), Math.toDegrees(ruta.dmMax)
				));
		jlDesviaM.setEnabled(true);
	}
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {

		Ruta re, rt;
		String fichero="Rutas/Parquin1";
		try {
			File file = new File(fichero);
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file));
			re=(Ruta)ois.readObject();
			rt=(Ruta)ois.readObject();
			ois.close();
			PanelExaminaRuta per=new PanelExaminaRuta(rt);
			JFrame ventana=new JFrame("PanelExaminaRuta");
			ventana.add(per);
			ventana.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
			ventana.pack();
			ventana.setVisible(true);
		} catch (IOException ioe) {
			System.err.println("Error al abrir el fichero " + fichero);
			System.err.println(ioe.getMessage());
		} catch (ClassNotFoundException cnfe) {
			System.err.println("Objeto leído inválido: " + cnfe.getMessage());            
		}     

		
	}


}
