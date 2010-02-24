/* */

package sibtra.ui;

import java.awt.BorderLayout;
import java.awt.Component;
import java.awt.Container;
import java.awt.FlowLayout;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;

import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JPopupMenu;
import javax.swing.JScrollPane;
import javax.swing.JSplitPane;
import javax.swing.JTabbedPane;
import javax.swing.KeyStroke;

import sibtra.log.PanelLoggers;
import sibtra.util.PanelDatos;

/**
 * Clase que tiene el esquema básico de las dos ventanas de la aplicación.
 * 
 * @author alberto
 *
 */
public class Ventanas  implements ActionListener  {
	
    
	/** La ventana principal ocupará casi toda la pantalla grande */
	public JFrame ventanaPrincipal=null;
	/** Ocupará toda la pantalla pequeña (táctil) */
	JFrame ventadaPeque=null;

	/** Barra de menu ventana principal */
	JMenuBar barraMenu;
	/** Menu de archivo dentro de la ventana principal */
	JMenu menuArchivo;
	/** Menu de acciones */
	JMenu menuAcciones;
	/** Boton de salir del menu de archivo */
	private JMenuItem miSalir;
	
	/** Panel de solapas derecho */
	JTabbedPane tbPanelDecho;
	
	/** Panel de solapas izquierdo */
	JTabbedPane tbPanelIzdo;

	/** Panel de los loggers */
	private PanelLoggers pmLog;
	
	/** Panel de la parte baja de la ventana principal */
	JPanel jpSur;
	
	/** Panel central de la ventana pequeña. Tiene @link {@link BoxLayout} */
	JPanel panelCentralPeque;
	
	private JMenu menuAyuda;
	private JMenuItem jmiAcercaDe;
	private JPanel panelResumen;

	/**
	 * Despues del constructor y de añadir todo lo que se quiera hay que invocar 
	 *   a {@link #muestraVentanas()} para que se termine la preparación y se muestren.
	 */
    public Ventanas() {
        
        ventanaPrincipal=new JFrame("Verdino");
        
        JPanel jpSur = new JPanel(new FlowLayout(3));
        ventanaPrincipal.getContentPane().add(jpSur, BorderLayout.SOUTH);

    	
    	//Solapas del lado izquierdo ===============================================
        tbPanelIzdo=new JTabbedPane();
    	//Panel datos numéricos se colacará a la izda del split panel


        //Panel con solapas para la parte derecha de la ventana principal =========================
        //  contendrá las gráficas.
        tbPanelDecho=new JTabbedPane();


        
        
//        tbPanelDecho.setPreferredSize(new Dimension(500,600));
//        tbPanelDecho.setMinimumSize(new Dimension(100,600));

    	//split panel en el centro de la ventana principal
        JSplitPane splitPanel=new JSplitPane(JSplitPane.HORIZONTAL_SPLIT
//        		,false  //si al mover la barra componentes se refrescan continuamente
        		,true  //si al mover la barra componentes se refrescan continuamente
        		,tbPanelIzdo
        		,tbPanelDecho
        );
        
        
		//Panel Resumen
		panelResumen=new JPanel( new GridLayout(0,4));
		añadePanel(panelResumen, "Resumen",true,false);

        ventanaPrincipal.getContentPane().add(splitPanel, BorderLayout.CENTER);

        //barra de menu
        barraMenu=new JMenuBar();
        //menu de archivo
        menuArchivo=new JMenu("Fichero");
        barraMenu.add(menuArchivo);

        miSalir=new JMenuItem("Salir");
        miSalir.setAccelerator(KeyStroke.getKeyStroke( KeyEvent.VK_Q,KeyEvent.CTRL_MASK));
        miSalir.addActionListener(this);
        //lo añadimos al menu al final
        
        //menu de Acciones
        menuAcciones=new JMenu("Acciones");
        barraMenu.add(menuAcciones);
        
        //menu de ayuda
        menuAyuda=new JMenu("Ayuda");
        jmiAcercaDe=new JMenuItem("Acerca de Verdino");
        jmiAcercaDe.addActionListener(this);
        menuAyuda.add(jmiAcercaDe);

        ventanaPrincipal.setJMenuBar(barraMenu); //ponemos barra en la ventana

        //Loggers en solapa con scroll panel
    	//Lo punemos cuando todos hayan apuntado sus loggers
        pmLog=new PanelLoggers();
        tbPanelDecho.add("Loggers",new JScrollPane(pmLog));

        //Menu con las acciones del los loggers
        JMenu menuLoggers=new JMenu("Loggers");
        barraMenu.add(menuLoggers);
        menuLoggers.add(pmLog.getAccionRefrescar());
        menuLoggers.add(pmLog.getAccionActivar());
        menuLoggers.add(pmLog.getAccionDesactivar());
        menuLoggers.add(pmLog.getAccionLimpiar());
        menuLoggers.add(pmLog.getAccionSalvar());
	
        
        //Mostramos la ventana principal con el tamaño y la posición deseada
        ventanaPrincipal.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        ventanaPrincipal.setUndecorated(true);
//        splitPanel.setDividerLocation(0.50); //La mitad para cada uno
        splitPanel.setDividerLocation(500); //Ajustamos para que no aparezca la barra a la dercha

        //La ventana Secundaria
        ventadaPeque=new JFrame("Tactil Verdino");
        panelCentralPeque=new JPanel();
        panelCentralPeque.setLayout(new BoxLayout(panelCentralPeque,BoxLayout.PAGE_AXIS));
        ventadaPeque.add(panelCentralPeque);
        
        ventadaPeque.setUndecorated(true); //para que no aparezcan el marco

    	// ponemos popups en las tabs
    	tbPanelDecho.addMouseListener(new MouseAdapter() {
    		public void mousePressed(MouseEvent me)
    		{
    			maybeShowPopup(me);
    		}

    		public void mouseReleased(MouseEvent me)
    		{
    			maybeShowPopup(me);
    		}
    	});
    	tbPanelIzdo.addMouseListener(new MouseAdapter() {
    		public void mousePressed(MouseEvent me)
    		{
    			maybeShowPopup(me);
    		}

    		public void mouseReleased(MouseEvent me)
    		{
    			maybeShowPopup(me);
    		}
    	});
    }

    void muestraVentanas() {

    	//Refrescamos los loggers
    	pmLog.getAccionRefrescar().actionPerformed(null);

    	//añadimos el boton de salir al final del menu de archivos
        menuArchivo.addSeparator(); //separador =============================
        menuArchivo.add(miSalir);
        
        //Menu de ayuda al final de la barra
        barraMenu.add(Box.createHorizontalGlue());
        barraMenu.add(menuAyuda);

        ventanaPrincipal.pack();
        ventanaPrincipal.setVisible(true);

        ventadaPeque.pack();
        ventadaPeque.setVisible(true);

        //Fijamos su tamaño y posición
        ventanaPrincipal.setBounds(0, 384, 1024, 742);
        //fijamos su tamaño y posición
        ventadaPeque.setBounds(0, 0, 640, 384);

    	
    }
    
    // Sacado de http://forums.sun.com/thread.jspa?forumID=257&threadID=372811
    private void maybeShowPopup(final MouseEvent me)
    {
    	JTabbedPane pest;
    	if (me.isPopupTrigger() 
    			&& (pest=(JTabbedPane)me.getSource()).getTabCount()>0
    			&& (pest.getSelectedIndex()<pest.getTabCount())
    	) {
    		JPopupMenu popup = new JPopupMenu();
    		JMenuItem item = new JMenuItem("Cambia de pestaña de lado");
    		popup.add(item);
    		item.addActionListener(new ActionListener() {
    			public void actionPerformed(ActionEvent e)
    			{
    				JTabbedPane tabbed = (JTabbedPane)me.getSource();
    				int i = tabbed.getSelectedIndex();
    				if(i>=tabbed.getTabCount()) return;
    				if(tabbed==tbPanelDecho) {
    					tbPanelIzdo.add(tabbed.getTitleAt(i),tabbed.getComponent(i));
    				} else {
    					tbPanelDecho.add(tabbed.getTitleAt(i),tabbed.getComponent(i));
    				}
    				//NO hace falta borrarla ??
    				//						tabbed.remove(i);
    			}
    		});
    		popup.show(me.getComponent(), me.getX(), me.getY());
    	}
    }    


    public void actionPerformed(ActionEvent e) {
		if(e.getSource()==miSalir) {
			Terminar();
		}
		if(e.getSource()==jmiAcercaDe) {
			JOptionPane.showMessageDialog(ventanaPrincipal, "VERDINO: ISAATC ULL");
		}
	}

    /** Metodo para terminar la ejecución */
    protected void Terminar() {
    	System.out.println("Terminamos ...");
		System.exit(0);
	}

    /**
     * Añade panel pasado a la solapa de uno de los lados
     * @param panel
     * @param titulo Nombre que llevará la solapa
     * @param enDerecho true se pone a la derecha, si no a la izda.
     * @param conScroll se mete panel dentro de scroll panel
     */
    public void añadePanel(JPanel panel, String titulo, boolean enDerecho, boolean conScroll) {
    	if(panel==null) return;

    	apuntaResumen(panel); //Si es panel de datos 
    	
    	JComponent ca=conScroll ? new JScrollPane(panel) : panel;
    		
    	if(enDerecho)
    		tbPanelDecho.add(titulo, ca );
    	else
    		tbPanelIzdo.add(titulo, ca );
    }

    /** Para el componente actual, y sus descendientes recursivamente, 
     * mira a ver si el {@link PanelDatos} y le apunta el {@link #panelResumen}  */
    private void apuntaResumen(Component ca) {
    	if(ca instanceof Container) {
    		if(ca instanceof PanelDatos) {
    			((PanelDatos)ca).setPanelResumen(panelResumen);
    		}
    		//aplicamos a sus hijos
    		for (Component comA: ((Container)ca).getComponents())
    			apuntaResumen(comA);
    	}
	}

	/** como {@link #añadePanel(JPanel, String, boolean, boolean)} con escroll */
    public void añadePanel(JPanel panel, String titulo, boolean enDerecho) {
    	añadePanel(panel, titulo, enDerecho, true);
    }
    
    /**
     * Añade panel pasado a la solapa del lado que tenga menos solapas
     * @param panel
     * @param titulo Nombre que llevará la solapa
     */
    public void añadePanel(JPanel panel, String titulo) {
    	añadePanel(panel, titulo, tbPanelDecho.getTabCount()<tbPanelIzdo.getTabCount());
    }
    
	public void quitaPanel(JPanel panel) {
		//Destruimos el panel
		invocaDestruir(panel); //para que quite etiquetas de resumne.
		//miramos si están en panel derecho
		JTabbedPane jtpAct=tbPanelDecho;
		for(int i=0; i<jtpAct.getComponentCount(); i++) {
			Component coa=jtpAct.getComponentAt(i);
			if(coa.equals(panel)) {
				jtpAct.remove(i);
				return;
			} else if(coa instanceof JScrollPane
					&& ((JScrollPane) coa).getViewport().getView().equals(panel)) {
				jtpAct.remove(i); 
				return;
			}
		}
		jtpAct=tbPanelIzdo;
		for(int i=0; i<jtpAct.getComponentCount(); i++) {
			Component coa=jtpAct.getComponentAt(i);
			if(coa.equals(panel)) {
				jtpAct.remove(i);
				return;
			} else if(coa instanceof JScrollPane
					&& ((JScrollPane) coa).getViewport().getView().equals(panel)) {
				jtpAct.remove(i); 
				return;
			}
		}
//		//miramos si están en panel izdo
//		jtpAct=tbPanelIzdo;
//		for(int i=0; i<jtpAct.getComponentCount(); i++) {
//			Component coa=jtpAct.getComponentAt(i);
//			if(coa.equals(panel))
//				jtpAct.remove(i);
//			else if(coa instanceof Container) 
//				for(Component subcoa: ((Container)coa).getComponents())
//					if(subcoa.equals(panel)) {
//						jtpAct.remove(i);
//						return;
//					}
//		}
	}

	
    /** Para el componente actual, y sus descendientes recursivamente, 
     * mira a ver si es {@link PanelDatos} y le invoca su método destruir()  */
    private void invocaDestruir(Component ca) {
    	if(ca instanceof Container) {
    		if(ca instanceof PanelDatos) {
    			((PanelDatos)ca).destruir();
    		}
    		//aplicamos a sus hijos
    		for (Component comA: ((Container)ca).getComponents())
    			invocaDestruir(comA);
    	}
	}


}
