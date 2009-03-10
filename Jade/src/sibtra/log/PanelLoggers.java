package sibtra.log;

import java.awt.Component;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Iterator;
import java.util.Vector;

import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JSpinner;
import javax.swing.JTabbedPane;
import javax.swing.JTable;
import javax.swing.SpinnerNumberModel;
import javax.swing.SwingUtilities;
import javax.swing.table.AbstractTableModel;

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
	private SpinnerNumberModel segActiva;
	private JButton limpiaButton;
	private JButton desactivaButton;
	private JTable talbaLoggers;

	/** Clase para apuntar en un vector el logger y si está seleccionado */
	class ApuntaLog {
		Logger la=null;
		boolean sel=false; //si esta seleccionado
		public ApuntaLog(Logger la,boolean sel) {
			this.la=la;
			this.sel=sel;
		}
	}
	/** Vector con los loger considerados y si están seleccionados */
	Vector<ApuntaLog> vecLA=null;

	static int COL_SEL=0;
	static int COL_NOM=1;
	static int COL_CLASS=2;
	static int COL_ACTIVA=3;
	static int COL_SIZE=4;
	static int COL_CAPA=5;

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
			talbaLoggers=new JTable(new modeloTablaLoggers());	
	        talbaLoggers.setPreferredScrollableViewportSize(new Dimension(500, 70));
//	        table.setAutoCreateRowSorter(true);

			panAct.add(new JScrollPane(talbaLoggers));

			panAct.add(Box.createRigidArea(new Dimension(0,5)));

			{ //Botones activacion, limpiado y salvado
				JPanel buttonPane = new JPanel();
				buttonPane.setMaximumSize(new Dimension(1200,60));
				buttonPane.setMinimumSize(new Dimension(700,60));
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

	/** Modelo de nuestra tabla de loggers */
    class modeloTablaLoggers extends AbstractTableModel {
    	
    	String[] nombColumnas=new String[6];
    	
    	
    	public modeloTablaLoggers() {
    		super();
    		//rellenamos array de titulos
        	nombColumnas[COL_SEL]="Sel.";
        	nombColumnas[COL_NOM]="Logger"; 
        	nombColumnas[COL_CLASS]="Clase"; 
        	nombColumnas[COL_ACTIVA]="Activado"; 
        	nombColumnas[COL_SIZE]="Tamaño"; 
        	nombColumnas[COL_CAPA]="Capacidad";
        	
    		//Apuntamos los logger disponibles
    		vecLA=new Vector<ApuntaLog>(LoggerFactory.vecLoggers.size());
    		for(Iterator<Logger> lit=LoggerFactory.vecLoggers.iterator(); lit.hasNext();) {
    			Logger la=lit.next();
    			vecLA.add(new ApuntaLog(la,true));
    		}
    	}
    	
        public String getColumnName(int col) {
            return nombColumnas[col].toString();
        }
        public int getRowCount() { return LoggerFactory.vecLoggers.size(); }
        public int getColumnCount() { return nombColumnas.length; }
        public Object getValueAt(int row, int col) {
        	// Sacamos los datos de vecLA
        	if(vecLA==null || vecLA.size()<=row)
        		return null;
        	Logger la=vecLA.get(row).la;
        	switch (col) {
        	case 0:
        		return vecLA.get(row).sel;
        	case 1:
        		return la.nombre;
        	case 2:
        		String no=la.objeto.getClass().getName();
        		return no.substring(no.lastIndexOf(".")+1);
        	case 3:
        		return (la.isActivo()?"ACTIVO":"    ");
        	case 4:
        		return (la.tiempos!=null)?la.tiempos.size():0;
        	case 5:
        		return (la.tiempos!=null)?la.tiempos.capacity():0;
        	default:
        		return null;
        	}
        }
        public void setValueAt(Object value, int row, int col) {
        	if(col!=0) return;
            vecLA.get(row).sel = (Boolean)value;
            fireTableCellUpdated(row, col);
        }

        public boolean isCellEditable(int row, int col) { 
        	return (col==0); //solo la columna 0 es editable 
        }
        public Class getColumnClass(int col) {
        	switch (col) {
        	case 0:
        		return Boolean.class;
        	case 1:
        	case 2:
        	case 3:
        		return String.class;
        	case 4:
        	case 5:
        		return Integer.class;
        	default:
        		return String.class;
        	}
        }

        /** Mira a ver si hay nuevos loggers y los añade. 
         * Como los loggers no se pueden borrar, basta con mirar el número 
         * */
        void añadeNuevosLoggers() {
        	if(LoggerFactory.vecLoggers==null)
        		return;
        	if(LoggerFactory.vecLoggers.size()==vecLA.size()) return; // no hay nuevos
        	//Comprobamos toda la lista de loggers por si han cambiado de posición
        	int ultimoAnt=vecLA.size();
        	int añadidos=0;
        	for(int i=0; i<LoggerFactory.vecLoggers.size(); i++) {
        		Logger la=LoggerFactory.vecLoggers.get(i);
        		//probamos primero en la posición correspondiente
        		if(i<vecLA.size() && vecLA.get(i).la==la) continue;
        		//si no lo buscamos en todos
        		boolean encontrado=false;
        		for(int j=0; !encontrado && j<vecLA.size(); j++)
        			encontrado=(la==vecLA.get(j).la);
        		if(encontrado) continue;
        		//si no se encontró lo añadimos
        		vecLA.add(new ApuntaLog(la,true));
        		añadidos++;
        	}
        	// Indicamos nuevas filas añadidas
        	if (añadidos>0)
        		fireTableRowsInserted(ultimoAnt+1, ultimoAnt+añadidos-1);
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
			for(int i=0; i<vecLA.size(); i++){
				if(!vecLA.get(i).sel) continue; // nos saltamso los no seleccionados
				Logger la=vecLA.get(i).la;
				if (ae.getSource()==activaButton) {
					la.activa(sgAct);
					System.out.println("Activando Logger:"+la.objeto.getClass().getName()+":"+la.nombre
							+" para "+sgAct+" segundos.");
//					actualizaCheckLoggers();
				} else if (ae.getSource()==limpiaButton) {
					la.clear();
					System.out.println("Limpiando Logger:"+la.objeto.getClass().getName()+":"+la.nombre
							+" para "+sgAct+" segundos.");
//					actualizaCheckLoggers();
				} else if (ae.getSource()==desactivaButton) {
					la.desactiva();
					System.out.println("Desactivando Logger:"+la.objeto.getClass().getName()+":"+la.nombre
							+" para "+sgAct+" segundos.");
//					actualizaCheckLoggers();
				}
			}
			talbaLoggers.repaint();
		}
		if(ae.getSource()==salvaButton) {
			String nombBase="Datos/PruPanLog";
			String nombCompleto=nombBase+new SimpleDateFormat("yyyyMMddHHmm").format(new Date())
			+".mat"
			;
			System.out.println("Escribiendo en Fichero "+nombCompleto);
			try {
				SalvaMATv4 smv4=new SalvaMATv4(nombCompleto);
				for(int i=0; i<vecLA.size(); i++){
					if(!vecLA.get(i).sel) continue; // nos saltamos los no seleccionados
					Logger la=vecLA.get(i).la;
					la.vuelcaMATv4(smv4);
					System.out.println("Volcando Logger:"+la.objeto.getClass().getName()+":"+la.nombre);
				}		
				smv4.close();
			} catch (IOException e) {
				// TODO Bloque catch generado automáticamente
				e.printStackTrace();
			}
		}
			
	}

	/** programamos la actualizacion del panel */
	public void repinta() {
		SwingUtilities.invokeLater(new Runnable() {
			public void run() {
				((modeloTablaLoggers)(talbaLoggers.getModel())).añadeNuevosLoggers();
				repaint();
			}
		});
	}
		
}
