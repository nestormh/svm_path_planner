package sibtra.log;

import java.awt.Component;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Iterator;

import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JSpinner;
import javax.swing.JTabbedPane;
import javax.swing.SpinnerNumberModel;

import sibtra.util.SalvaMATv4;

/**
 * Panel que permitirá gestionar los loggers:
 * seleccionar los loggers a activar, activarlos, salvar datos a fichero, etc.
 * @author alberto
 *
 */
public class PanelLoggers extends JTabbedPane implements ActionListener {

	private JButton activaButton;
	private JButton salvaButton;
	private JPanel panLoggers;
	private SpinnerNumberModel segActiva;
	private JButton limpiaButton;
	private JButton desactivaButton;

	public PanelLoggers() {
		super(JTabbedPane.TOP, JTabbedPane.WRAP_TAB_LAYOUT);
		
		{ //primera pestaña para activar
			JPanel panAct=new JPanel();
			panAct.setLayout(new BoxLayout(panAct,BoxLayout.PAGE_AXIS));
			//arriba el titulo
			JLabel titulo=new JLabel("Loggers Definidos:");
			titulo.setAlignmentX(Component.LEFT_ALIGNMENT);
			panAct.add(titulo);
			

			panAct.add(Box.createRigidArea(new Dimension(0,5)));
			
			//Donde seleccionar los loggers
			panLoggers=new JPanel();	
			panLoggers.setLayout(new BoxLayout(panLoggers,BoxLayout.PAGE_AXIS));
			actualizaCheckLoggers();
			panAct.add(new JScrollPane(panLoggers));

			panAct.add(Box.createRigidArea(new Dimension(0,5)));

			{ //Botones activacion, limpiado y salvado
				JPanel buttonPane = new JPanel();
				buttonPane.setLayout(new BoxLayout(buttonPane, BoxLayout.LINE_AXIS));
				buttonPane.setBorder(BorderFactory.createEmptyBorder(0, 10, 10, 10));

				buttonPane.add(Box.createHorizontalGlue());

				limpiaButton=new JButton("Limpiar");
				limpiaButton.addActionListener(this);
				buttonPane.add(limpiaButton);
				
				buttonPane.add(Box.createRigidArea(new Dimension(10, 0)));
				
				desactivaButton=new JButton("Desactiva");
				desactivaButton.addActionListener(this);
				buttonPane.add(desactivaButton);
				
				buttonPane.add(Box.createRigidArea(new Dimension(10, 0)));
				
				activaButton=new JButton("Activar");
				activaButton.addActionListener(this);
				buttonPane.add(activaButton);
				
				buttonPane.add(new JLabel(" para "));
				segActiva=new SpinnerNumberModel(300,10,3600,5);
				buttonPane.add(new JSpinner(segActiva));
				buttonPane.add(new JLabel(" sg."));
				
				
				salvaButton=new JButton("Salva");
				salvaButton.addActionListener(this);
				buttonPane.add(salvaButton);
				panAct.add(buttonPane);
			}
			//Añadimos panel a la pestaña
			addTab("Activacion", panAct);			
		}
	}

	/** Obtiene la lista de logges y añade checkbox para cada uno con su nombre */
	void actualizaCheckLoggers() {
		if(LoggerFactory.vecLoggers==null)
			return;
		for(Iterator<Logger> lit=LoggerFactory.vecLoggers.iterator(); lit.hasNext();) {
			Logger la=lit.next();
			String tituloLa=la.objeto.getClass().getName()+":"+la.nombre
			+(la.isActivo()?" ACTIVO ":" no activo ")
			+(la.tiempos!=null?la.tiempos.size()+"/"+la.tiempos.capacity():"0/0")
			;
			Component[] ca=panLoggers.getComponents();
			boolean encontrado=false;
			for(int i=0; i<ca.length; i++){
				Logger lapan;
				JCheckBox jcba;
				if (
						(ca[i] instanceof JCheckBox) 
						&& ((lapan=(Logger)(jcba=(JCheckBox)ca[i]).getClientProperty("Logger"))!=null)
						&& (la==lapan)
				) { //ya hay un checkbox para este logger, lo actulizamos
					jcba.setText(tituloLa);
					encontrado=true;
				}
			}
			if(!encontrado) { //Si no se encontró, lo creamos nuevo y lo añadimos
				JCheckBox jcb=new JCheckBox(tituloLa);
				jcb.setSelected(true); //todos seleccionados por defecto
				jcb.putClientProperty("Logger", la); //apuntamos el loger que le corresponde
				panLoggers.add(jcb);
			}
		}
	}

	public void actionPerformed(ActionEvent ae) {
		//Para el boton Activa
		if(ae.getSource()==activaButton 
				|| ae.getSource()==limpiaButton
				|| ae.getSource()==desactivaButton
		) {
			//actuamos sobre aquellos loggers seleccionados
			int sgAct=segActiva.getNumber().intValue();
			Component[] ca=panLoggers.getComponents();
			for(int i=0; i<ca.length; i++){
				Logger la;
				JCheckBox jcba;
				if (
						(ca[i] instanceof JCheckBox) 
						&& ((la=(Logger)(jcba=(JCheckBox)ca[i]).getClientProperty("Logger"))!=null)
						&& (jcba.isSelected())
				) if (ae.getSource()==activaButton) {
					la.activa(sgAct);
					System.out.println("Activando Logger:"+la.objeto.getClass().getName()+":"+la.nombre
							+" para "+sgAct+" segundos.");
					actualizaCheckLoggers();
				} else if (ae.getSource()==limpiaButton) {
					la.clear();
					System.out.println("Limpiando Logger:"+la.objeto.getClass().getName()+":"+la.nombre
							+" para "+sgAct+" segundos.");
					actualizaCheckLoggers();
				} else if (ae.getSource()==desactivaButton) {
					la.desactiva();
					System.out.println("Desactivando Logger:"+la.objeto.getClass().getName()+":"+la.nombre
							+" para "+sgAct+" segundos.");
					actualizaCheckLoggers();
				}
			}
		}
		if(ae.getSource()==salvaButton) {
			String nombBase="Datos/PruPanLog";
			String nombCompleto=nombBase+new SimpleDateFormat("yyyyMMddHHmm").format(new Date())
			+".mat"
			;
			System.out.println("Escribiendo en Fichero "+nombCompleto);
			try {
				SalvaMATv4 smv4=new SalvaMATv4(nombCompleto);
				Component[] ca=panLoggers.getComponents();
				for(int i=0; i<ca.length; i++){
					Logger la;
					JCheckBox jcba;
					if (
							(ca[i] instanceof JCheckBox) 
							&& ((la=(Logger)(jcba=(JCheckBox)ca[i]).getClientProperty("Logger"))!=null)
							&& (jcba.isSelected())
					) {
						la.vuelcaMATv4(smv4);
						System.out.println("Volcando Logger:"+la.objeto.getClass().getName()+":"+la.nombre);
					}
				}		
				smv4.close();
			} catch (IOException e) {
				// TODO Bloque catch generado automáticamente
				e.printStackTrace();
			}
		}
			
	}
	
	
		
}
