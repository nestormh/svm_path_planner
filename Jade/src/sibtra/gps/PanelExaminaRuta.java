/**
 * 
 */
package sibtra.gps;

import java.awt.BorderLayout;
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
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;


/**
 * {@link JPanel} para examinar el contenido de una ruta.
 * El ínidice del punto seleccionado va de 0 a NumPuntos -1
 * @author alberto
 *
 */
@SuppressWarnings("serial")
public class PanelExaminaRuta extends JPanel implements ActionListener, ChangeListener {
	
	private PanelMuestraGPSData PanelPto;
	
	Ruta ruta;

	private JButton jbPrimero;
	private JButton jbUltimo;

	JSpinner jsDato;

	private SpinnerNumberModel spmPuntoActual;

	private JLabel jlDeMaximo;

	private JLabel jlDesviaM;

	SpinnerNumberModel spUmbral;

	private JSpinner jspUmbral;

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
		
		{ //controles parte Alta
			JPanel jpInf=new JPanel();
			
			jbPrimero=new JButton("<<");
			jbPrimero.setToolTipText("Primero");
			jbPrimero.addActionListener(this);
			jpInf.add(jbPrimero);
						
			spmPuntoActual=new SpinnerNumberModel(0,0,100000,1);
			jsDato=new JSpinner(spmPuntoActual);
			jsDato.setSize(150, jsDato.getHeight());
			jsDato.addChangeListener(this);
			jpInf.add(jsDato);
			
			jlDeMaximo=new JLabel(" de ???? ");
			jpInf.add(jlDeMaximo);
						
			jbUltimo=new JButton(">>");
			jbUltimo.setToolTipText("Ultimo");
			jbUltimo.addActionListener(this);
			jpInf.add(jbUltimo);
			
			//Para la desviación magnética
			jlDesviaM=new JLabel("Desviación magnetica ##.##º con umbral ");
			jpInf.add(jlDesviaM);
			spUmbral=new SpinnerNumberModel(10.00,0,180,0.25);
			spUmbral.addChangeListener(this);
			jspUmbral=new JSpinner(spUmbral);
			jpInf.add(jspUmbral);
			jspUmbral.setEnabled(false);
			
			add(jpInf,BorderLayout.PAGE_START);
		}
		setRuta(ra);
	}
	
	/** Cambia la ruta presentada como {@link #setRuta(Ruta, boolean)} pero sin mantener posición */ 
	public void setRuta(Ruta ra) {
		setRuta(ra,false); //por defecto no se mantiene la posición
	}
	
	/** Cambia la ruta presentada
	 * @param ra nueva ruta a presentar
	 * @param mantienePosicion si se trata de dejar el mismo índice para el punto seleccinado. 
	 * Útil si es la misma ruta a la que se ha quitado un punto
	 */
	public void setRuta(Ruta ra, boolean mantienePosicion) {
		ruta=ra;
		if(ruta==null || ruta.getNumPuntos()==0) {
			jbPrimero.setEnabled(false);
			jsDato.setEnabled(false);
			jbUltimo.setEnabled(false);
			jlDesviaM.setEnabled(false);
			jspUmbral.setEnabled(false);
		} else {
			jbPrimero.setEnabled(true);
			jsDato.setEnabled(true);
			jbUltimo.setEnabled(true);
			int indMax=ruta.getNumPuntos()-1;
			jlDeMaximo.setText(String.format(" de %d ", indMax));
			if(!mantienePosicion)
				spmPuntoActual.setValue(0); //nos ponemos al principio
			else //dejamos como está salvo que estemos fuera de rango.
				if((Integer)spmPuntoActual.getValue()>indMax)
					spmPuntoActual.setValue(indMax);
			spmPuntoActual.setMaximum(indMax);
			spUmbral.setValue(Math.toDegrees(ra.getUmbralDesviacion()));
			actualizaDM();
			jspUmbral.setEnabled(true);
		}
	}

	/* (non-Javadoc)
	 * @see java.awt.event.ActionListener#actionPerformed(java.awt.event.ActionEvent)
	 */
	public void actionPerformed(ActionEvent ae) {
		if(ae.getSource()==jbPrimero) {
			jsDato.setValue(0);
		} else if(ae.getSource()==jbUltimo) {
			jsDato.setValue(ruta.getNumPuntos()-1);
		}
	}

	/* (non-Javadoc)
	 * @see javax.swing.event.ChangeListener#stateChanged(javax.swing.event.ChangeEvent)
	 */
	public void stateChanged(ChangeEvent ce) {
		if(ce.getSource()==jsDato) {
			GPSData npto=ruta.getPunto((Integer)jsDato.getValue());
			PanelPto.actualizaPunto(npto);
//			System.out.println(npto.getCadenaNMEA());
			
		}
		if(ce.getSource()==spUmbral) {
			actualizaDM();
		}
	}

	/** Fija el índice del punto seleccionado */
	public void setIndice(int ind) {
		if(ind<0 || ind>=ruta.getNumPuntos())
			throw new IllegalArgumentException("Indice a fijar ("+ind+") fuera de rango");
		jsDato.setValue(ind);
	}

	/** @return el indice del punto seleccionado */
	public int getIndice() {
		return (Integer)jsDato.getValue();
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
